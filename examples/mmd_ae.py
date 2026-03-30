from itertools import cycle
from collections.abc import Callable
from logging import warning
import os
import warnings
import dwave_networkx as dnx
import networkx as nx
import numpy as np
import torch
from torch import nn
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms.v2 import Compose, ToDtype, ToImage
from torchvision.utils import make_grid, save_image

from dwave.plugins.torch.models.boltzmann_machine import (
    GraphRestrictedBoltzmannMachine as GRBM,
)
from dwave.plugins.torch.nn.functional import bit2spin_soft, spin2bit_soft
from dwave.system import DWaveSampler
from minorminer.subgraph import find_subgraph

from dwave.system.composites import FixedEmbeddingComposite
from dwave.preprocessing.composites import SpinReversalTransformComposite
from dwave.experimental.automorphism.automorphism_composite import AutomorphismComposite


class RadialBasisFunction(nn.Module):

    def __init__(
        self,
        n_kernels: int = 5,
        mul_factor: float = 2.0,
        bandwidth: torch.Tensor | float | None = None,
    ) -> None:
        """Initializes the Radial Basis Function (RBF) kernel module.

        Args:
            n_kernels: The number of RBF kernels to use. Each kernel will have a different bandwidth, determined by the mul_factor.
            mul_factor: The multiplicative factor that determines the bandwidth of each RBF kernel. The bandwidths will be spaced geometrically, with the middle kernel having a bandwidth of `bandwidth` (if provided) or the average pairwise distance (if `bandwidth` is None).
            bandwidth: The bandwidth for the RBF kernels. If None, the bandwidth will be set to the average pairwise distance between samples in the input tensor during the forward pass.
        """
        super().__init__()
        bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.register_buffer("bandwidth_multipliers", bandwidth_multipliers)
        self.bandwidth = bandwidth

    def get_bandwidth(self, l2_dist: torch.Tensor) -> torch.Tensor | float:
        """Computes the bandwidth for the RBF kernels based on the input pairwise L2 distances.

        Args:
            l2_dist: A tensor of shape (n, n) containing the pairwise L2 distances between samples in the input tensor.

        Returns:
            A tensor or float representing the bandwidth for the RBF kernels.
        """
        if self.bandwidth is None:
            n = l2_dist.shape[0]
            avg = l2_dist.sum() / (n**2 - n)  # (diagonal is zero)
            return avg

        return self.bandwidth

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass through the RBF kernel module.

        Computes the pairwise RBF kernel values for the input tensor X.

        Args:
            X: A tensor of shape (n, d) representing n samples with d features.

        Returns:
            A tensor of shape (n, n) containing the pairwise RBF kernel values.
        """
        l2 = torch.cdist(X, X) ** 2
        bandwidth = self.get_bandwidth(l2.detach()) * self.bandwidth_multipliers
        res = torch.exp(-l2.unsqueeze(0) / bandwidth.reshape(-1, 1, 1)).sum(dim=0)
        return res


class MMDLoss(nn.Module):
    def __init__(self, kernel: nn.Module = RadialBasisFunction()) -> None:
        """Initialize the MMD loss module.

        Args:
            kernel: A kernel function to compute the MMD.
        """
        super().__init__()
        self.kernel = kernel

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Computes the maximum mean discrepancy (MMD) between two sets of
        samples X and Y using the specified kernel.

        Args:
            X: A tensor of shape (n, d) representing the first set of samples.
            Y: A tensor of shape (m, d) representing the second set of samples.

        Returns:
            A scalar tensor representing the maximum mean discrepancy between
            X and Y.
        """
        K = self.kernel(torch.vstack([X.flatten(1), Y.flatten(1)]))
        n = X.shape[0]
        m = Y.shape[0]
        XX = (K[:n, :n].sum() - K[:n, :n].trace()) / (n * (n - 1))
        YY = (K[n:, n:].sum() - K[n:, n:].trace()) / (m * (m - 1))
        XY = K[:n, n:].mean()
        mmd = XX - 2 * XY + YY
        return mmd


class SkipLinear(nn.Module):
    def __init__(self, din: int, dout: int) -> None:
        """Initialize a skip connection for a linear layer.

        Args:
            din: The input dimension.
            dout: The output dimension.
        """
        super().__init__()
        self.linear = nn.Linear(din, dout, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the layer.

        Args:
            x: A tensor of shape (batch_size, din) representing the input.

        Returns:
            A tensor of shape (batch_size, dout) representing the output.
        """
        return self.linear(x)


class LinearBlock(nn.Module):
    def __init__(self, din: int, dout: int, sn: bool, p: float, bias: bool) -> None:
        """Initialize a linear block with skip connections.

        Args:
            din: The input dimension.
            dout: The output dimension.
            sn: Whether to use spectral normalization.
            p: The dropout probability.
            bias: Whether to include a bias term in the linear layers.
        """
        super().__init__()
        self.skip = SkipLinear(din, dout)
        linear_1 = nn.Linear(din, dout, bias)
        linear_2 = nn.Linear(dout, dout, bias)
        self.block = nn.Sequential(
            nn.LayerNorm(din),
            linear_1,
            nn.Dropout(p),
            nn.ReLU(),
            nn.LayerNorm(dout),
            linear_2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the linear block with skip connections.

        Args:
            x: Input tensor of shape (batch_size, din).

        Returns:
            Output tensor of shape (batch_size, dout).
        """
        return self.block(x) + self.skip(x)


class ConvolutionBlock(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int], cout: int) -> None:
        """Initialize a convolutional block with skip connections.

        Args:
            input_shape: Shape of the input tensor (channels, height, width).
            cout: Number of output channels.
        """
        super().__init__()
        input_shape = tuple(input_shape)
        cin, hx, wx = input_shape
        if hx != wx:
            raise NotImplementedError("TODO")

        self.input_shape = tuple(input_shape)
        self.cin = cin
        self.cout = cout

        self.block = nn.Sequential(
            nn.LayerNorm(input_shape),
            nn.Conv2d(cin, cout, 3, 1, 1),
            nn.ReLU(),
            nn.LayerNorm((cout, hx, wx)),
            nn.Conv2d(cout, cout, 3, 1, 1),
        )
        self.skip = SkipConv2d(cin, cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the convolutional block with skip connections.

        Args:
            x: Input tensor of shape (batch_size, cin, height, width).

        Returns:
            Output tensor of shape (batch_size, cout, height, width).
        """
        return self.block(x) + self.skip(x)


class SkipConv2d(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        """Initialize a skip connection with a 1x1 convolution.

        Args:
            cin: Number of input channels.
            cout: Number of output channels.
        """
        super().__init__()
        self.skip = nn.Conv2d(cin, cout, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the skip connection.

        Args:
            x: A tensor of shape (batch_size, cin, height, width) representing the input.

        Returns:
            A tensor of shape (batch_size, cout, height, width) representing the output.
        """
        return self.skip(x)


class ConvolutionNetwork(nn.Module):
    def __init__(self, channels: list[int], input_shape: tuple[int, int, int]) -> None:
        """Initialize a convolutional network with skip connections.

        Args:
            channels: List of output channels for each convolutional block.
            input_shape: Shape of the input tensor (channels, height, width).
        """
        super().__init__()
        channels = channels.copy()
        input_shape = tuple(input_shape)
        cx, hx, wx = input_shape
        if hx != wx:
            raise NotImplementedError("TODO")
        self.channels = channels
        self.cin = cx
        self.cout = self.channels[-1]
        self.input_shape = input_shape

        channels_in = [cx] + channels[:-1]
        self.blocks = nn.Sequential()
        for cin, cout in zip(channels_in, channels):
            self.blocks.append(ConvolutionBlock((cin, hx, wx), cout))
            self.blocks.append(nn.ReLU())
        self.blocks.pop(-1)
        self.skip = SkipConv2d(cx, cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the convolutional network with skip connections.

        Args:
            x: Input tensor of shape (batch_size, cin, height, width).

        Returns:
            Output tensor of shape (batch_size, cout, height, width).
        """
        x = self.blocks(x) + self.skip(x)
        return x


class FullyConnectedNetwork(nn.Module):
    def __init__(
        self,
        din: int,
        dout: int,
        depth: int,
        sn: bool,
        p: float,
        bias: bool = True,
    ) -> None:
        """Fully connected network with skip connections.

        Args:
            din: Input dimension.
            dout: Output dimension.
            depth: Number of layers.
            sn: Whether to use spectral normalization.
            p: Dropout probability.
            bias: Whether to include a bias term.
        """
        super().__init__()
        if depth == 1:
            raise ValueError("Depth must be at least 2.")
        self.skip = SkipLinear(din, dout)
        big_d = max(din, dout)
        dims = [big_d] * (depth - 1) + [dout]
        self.blocks = nn.Sequential()
        for d_in, d_out in zip([din] + dims[:-1], dims):
            self.blocks.append(LinearBlock(d_in, d_out, sn, p, bias))
            self.blocks.append(nn.Dropout(p))
            self.blocks.append(nn.ReLU())
        # Remove the last ReLU and Dropout
        self.blocks.pop(-1)
        self.blocks.pop(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through fully connected network with skip connections.

        Args:
            x: Input tensor of shape (batch_size, din).

        Returns:
            Output tensor of shape (batch_size, dout).
        """
        return self.blocks(x) + self.skip(x)


def straight_through_bitrounding(fuzzy_bits: torch.Tensor) -> torch.Tensor:
    """Applies straight-through estimation for bit rounding.

    Args:
        fuzzy_bits: Tensor with values in [0, 1].

    Returns:
        Tensor with values rounded to 0 or 1.

    Raises:
        ValueError: If any value in fuzzy_bits is outside the range [0, 1].
    """
    if not ((fuzzy_bits >= 0) & (fuzzy_bits <= 1)).all():
        raise ValueError(f"Inputs should be in [0, 1]: {fuzzy_bits}")
    bits = fuzzy_bits + (fuzzy_bits.round() - fuzzy_bits).detach()
    return bits


class StraightThroughTanh(nn.Module):
    def __init__(self) -> None:
        """Initialize the straight-through tanh binarization module.

        The forward pass applies tanh, maps spins to bits, rounds with a
        straight-through estimator, and maps the result back to spins.
        """
        super().__init__()
        self.hth = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the binarizing layer.

        Args:
            x: Real-valued tensor.

        Returns:
            Binarized tensor with values in {-1, 1}.
        """
        fuzzy_spins = self.hth(x)
        fuzzy_bits = spin2bit_soft(fuzzy_spins)
        bits = straight_through_bitrounding(fuzzy_bits)
        spins = bit2spin_soft(bits)
        return spins


def zephyr_subgraph(G: nx.Graph, zephyr_m: int) -> nx.Graph:
    """Create a Zephyr subgraph of reduced m parameter.

    The first zephyr sublattice is used. The subgraph is
    subject to the same yield as the source graph.
    More sophisticated tools that map to increase the yield are in
    development.

    Args:
        G: A Zephyr graph.
        zephyr_m: The shape parameter of a zephyr graph, also called number
            of rows. A subgraph of this scale is found within G, and
            returned.

    Returns:
        The subgraph as a network Graph object.
    """
    assert (
        zephyr_m <= G.graph["rows"]
    ), "zephyr_m must be less than or equal to the number of rows in G"

    Z_m = dnx.zephyr_graph(zephyr_m)
    zsm = next(dnx.zephyr_sublattice_mappings(Z_m, G))
    S = G.subgraph([zsm(z) for z in Z_m])
    original_m = S.graph["rows"]
    if original_m == zephyr_m:
        return G.copy()
    S.graph = G.graph.copy()
    S.graph["rows"] = zephyr_m
    S.graph["columns"] = zephyr_m
    S.graph["name"] = S.graph["name"].replace(f"({original_m},", f"({zephyr_m},")
    S.graph["name"] = f'{S.graph["name"]}-subgraph of {G.graph["name"]}'
    return S


def zephyr_subgraph_t(G: nx.Graph, zephyr_t: int) -> nx.Graph:
    """Create a Zephyr subgraph with a reduced tile parameter.

    The subgraph is subject to the same yield as the source graph.
    More sophisticated tools that map to increase the yield are in
    development.

    Args:
        G: A Zephyr graph.
        zephyr_t: The tile parameter of a zephyr graph.

    Returns:
        A subgraph of the original Zephyr graph containing the specified number of tiles.
    """
    assert (
        zephyr_t <= G.graph["tile"]
    ), "zephyr_t must be less than or equal to the tile parameter of G"

    zc = dnx.zephyr_coordinates(m=G.graph["rows"], t=G.graph["tile"])
    return G.subgraph([g for g in G if zc.linear_to_zephyr(g)[2] < zephyr_t])


@torch.compile
class Autoencoder(nn.Module):

    def __init__(self, shape: tuple[int, int, int], n_bits: int) -> None:
        """Initialize the autoencoder model.

        Args:
            shape: shape of the input images, as (channels, height, width).
            n_bits: number of bits for the latent representation.
        """
        super().__init__()
        _, h, w = shape
        chidden = 1
        depth_fcnn = 3
        depth_cnn = 3
        dropout = 0.0
        self.encoder = nn.Sequential(
            ConvolutionNetwork([chidden] * depth_cnn, shape),
            nn.Flatten(),
            FullyConnectedNetwork(chidden * h * w, n_bits, depth_fcnn, False, dropout),
        )
        self.binarizer = StraightThroughTanh()
        self.decoder = nn.Sequential(
            FullyConnectedNetwork(n_bits, chidden * h * w, depth_fcnn, False, dropout),
            nn.Unflatten(1, (chidden, h, w)),
            ConvolutionNetwork([chidden] * (depth_cnn - 1) + [1], (chidden, h, w)),
            # nn.Sigmoid()
        )

    def decode(self, q: torch.Tensor) -> torch.Tensor:
        """Decode a latent representation back to the input space.

        Args:
            q: latent representation.

        Returns:
            Reconstructed input from the latent representation.
        """
        logits = self.decoder(q)
        return logits

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the autoencoder.

        Args:
            x: Input tensor.

        Returns:
            A tuple containing the latent representation, the binarized latent
            representation, and the reconstructed input.
        """
        soft_spins = self.encoder(x)
        spins = self.binarizer(soft_spins)
        logits = self.decode(spins)
        return soft_spins, spins, logits


def collect_stats(
    model: Autoencoder,
    grbm: GRBM,
    x: torch.Tensor,
    q: torch.Tensor,
    compute_mmd: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    compute_pkl: Callable[
        [GRBM, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
    ],
) -> dict[str, torch.Tensor]:
    """Collects statistics for the autoencoder and GRBM.

    Args:
        model: The Autoencoder.
        grbm: A graph restricted Boltzmann machine.
        x: Input tensor.
        q: Latent representation tensor from the GRBM model.
        compute_mmd: Function to compute the maximum mean discrepancy between two tensors.
        compute_pkl: Function to compute the KL divergence between the GRBM and the latent representation.

    Returns:
        A dictionary containing the computed statistics.
    """
    soft_spins, spins, logits = model(x)
    stats = {
        "quasi": grbm.quasi_objective(spins.detach(), q),
        "mse": nn.functional.mse_loss(logits.sigmoid(), x),
        "bce": nn.functional.binary_cross_entropy_with_logits(logits, x),
        "mmd": compute_mmd(spins, q),
        "pkl": compute_pkl(grbm, soft_spins, spins, q),
    }
    return stats


def get_dataset(bs: int, data_dir: str = "/tmp/") -> tuple[DataLoader, DataLoader]:
    """Loads the MNIST dataset and returns data loaders for training and testing.

    Args:
        bs: Batch size.
        data_dir: Directory to download the dataset.

    Returns:
        A tuple containing the training and testing data loaders.
    """
    transforms = Compose([ToImage(), ToDtype(torch.float32, scale=True)])
    train_kwargs = dict(root=data_dir, download=True)
    transforms = Compose([transforms, lambda x: 1 - x])
    data_train = MNIST(transform=transforms, **train_kwargs)
    train_loader = DataLoader(data_train, bs, True)
    data_test = MNIST(transform=transforms, **train_kwargs, train=False)
    test_loader = DataLoader(data_test, bs, True)
    return train_loader, test_loader


def save_viz(
    model: Autoencoder,
    x: torch.Tensor,
    q: torch.Tensor,
    title: str = "",
) -> None:
    """Saves visualizations of the input, generated, and reconstructed images.

    Args:
        model: The Autoencoder.
        x: Input tensor.
        q: Latent representation tensor.
        title: Prefix used for generated image files.
    """
    bs = min(x.shape[0], 500)
    rows = int(bs**0.5)
    with torch.no_grad():
        # Save images
        xgen = model.decode(q[:bs]).sigmoid()
        xuni = model.decode(bit2spin_soft(torch.randint_like(q[:bs], 2))).sigmoid()
        _, _, logits = model(x[:bs])
        logits = logits.sigmoid()
        xgrid = make_grid(x[:bs], rows, pad_value=1)
        xgengrid = make_grid(xgen, rows, pad_value=1)
        xunigrid = make_grid(xuni, rows, pad_value=1)
        logits_grid = make_grid(logits, rows, pad_value=1)
        save_image(xgrid, f"{title}x.png")
        save_image(xgengrid, f"{title}xgen.png")
        save_image(xunigrid, f"{title}xuni.png")
        save_image(logits_grid, f"{title}xhat.png")


def node_coloring(G: nx.Graph) -> dict[int, int]:
    """Computes a node coloring to accelerate find_subgraph

    Args:
        G: A networkx graph.

    Returns:
        A dictionary mapping each node in G to an integer color.
    """
    if G.graph["family"] == "zephyr":
        to_coord = dnx.zephyr_coordinates(
            m=G.graph["rows"], t=G.graph["tile"]
        ).linear_to_zephyr
        co_index = 0
    elif G.graph["family"] == "pegasus":
        to_coord = dnx.pegasus_coordinates(
            m=G.graph["rows"], t=G.graph["tile"]
        ).linear_to_pegasus
        co_index = 0
    elif G.graph["family"] == "chimera":
        to_coord = dnx.chimera_coordinates(
            m=G.graph["rows"], t=G.graph["tile"]
        ).linear_to_chimera
        co_index = 2
    else:
        raise ValueError("Unknown case")
    return {n: to_coord(n)[co_index] for n in T.nodes}


def get_qpu_model_grbm(
    solver: str,
    device: str,
    m: int = 5,
    t: int = 3,
    dnx_family: str = "zephyr",
    timeout: int = 60,
    allow_incomplete_yield: bool = False,
    use_srts: bool = False,
    use_automorphisms: bool = False,
    orientation_hint: bool = True,
) -> tuple[DWaveSampler, Autoencoder, GRBM]:
    """Sets up the QPU, GRBM, and Autoencoder model.

    Args:
        solver: The D-Wave solver name.
        device: The device to run the model on, typically "cuda" or "cpu".
        m: Rows and columns of a small Chimera graph
        t: Tile parameter of a small Chimera graph
        dnx_family: The family of D-Wave hardware to target for the subgraph embedding. This is used to determine the structure of the Chimera graph to embed, which should be compatible with the target hardware. For example, "zephyr" would indicate that we want to embed a Zephyr subgraph, which is a specific type of Chimera graph with certain connectivity properties.
        timeout: timeout for chimera graph search.
    Returns:
        A tuple containing the QPU sampler, Autoencoder model, and GRBM.
    """
    # Set up QPU and QPU parameters
    qpu = DWaveSampler(solver=solver)
    # Instantiate model
    T = qpu.to_networkx_graph()
    if dnx_family == "zephyr":
        S = dnx.zephyr_graph(m)
    elif dnx_family == "pegasus":
        S = dnx.pegasus_graph(m)
    elif dnx_family == "chimera":
        S = dnx.chimera_graph(m, m, t)
    else:
        raise ValueError(f"Unknown dnx_family: {dnx_family}")
    if orientation_hint:
        node_colors = (node_coloring(S), node_coloring(T))
    else:
        node_colors = None

    emb = find_subgraph(
        S, T, timeout=timeout, as_embedding=True
    )  # TO DO: add orientation hinting
    if len(emb) < S.number_of_nodes():
        if not allow_incomplete_yield:
            raise RuntimeError(
                f"Failed to find an embedding of the {dnx_family} graph "
                f"with m={m} and t={t} within the timeout {timeout}s."
                "Consider a simpler graph, smaller m and/or t, or larger timeout."
            )

        warnings.warn("legacy method, requires improvement")
        # G = zephyr_subgraph_t(zephyr_subgraph(qpu.to_networkx_graph(), m), t)  # Old
        print(
            "S num edges and vars targetted", S.number_of_edges(), S.number_of_nodes()
        )
        S = zephyr_subgraph_t(zephyr_subgraph(T, m), t)  # Old
        # S = T.edge_subgraph(S.edges) # Only works for coordinated cases.
        print("S num edges and vars realized", S.number_of_edges(), S.number_of_nodes())
        emb = {n: (n,) for n in S.nodes}
    nodes = list(S.nodes)
    edges = list(S.edges)
    grbm = GRBM(nodes, edges).to(device)
    # grbm.linear.data[:] = 0
    # grbm.quadratic.data[:] = 0
    model = Autoencoder((1, 28, 28), grbm.n_nodes).to(device)

    sampler = FixedEmbeddingComposite(qpu, emb)
    if use_srts:
        sampler = SpinReversalTransformComposite(sampler)
    if use_automorphisms:
        sampler = AutomorphismComposite(sampler, G=S)
    for key in ["h_range", "j_range"]:
        sampler.properties[key] = qpu.properties[key]  # type: ignore
    return sampler, model, grbm


def compute_pkl(
    grbm: GRBM,
    logits_data: torch.Tensor,
    spins_data: torch.Tensor,
    spins_model: torch.Tensor,
) -> torch.Tensor:
    """Computes the pseudo-Kullback-Leibler divergence.

    Args:
        grbm: A graph restricted Boltzmann machine.
        logits_data: The pre-sigmoid outputs of the autoencoder decoder for the input data.
        spins_data: The binarized latent representation for the input data.
        spins_model: The binarized latent representation for the model samples.

    Returns:
        A scalar tensor representing the pseudo-KL divergence between the GRBM and the latent representation.
    """
    probabilities = torch.sigmoid(logits_data)
    entropy = torch.nn.functional.binary_cross_entropy_with_logits(
        logits_data, probabilities
    )
    # bce = p(log(q)) + (1-p) log(1-q)
    cross_entropy = grbm.quasi_objective(spins_data, spins_model)
    pkl = cross_entropy - entropy
    return pkl


def run(
    *,
    title: str,
    loss_fn: str,
    solver: str,
    stop_grbm: int,
    num_reads: int,
    annealing_time: float,
    alpha: float,
    num_steps: int,
    device: str = "cuda",
    seed: int | None = None,
    m: int,
    t: int,
    use_srts: bool = False,
    use_automorphisms: bool = False,
    allow_incomplete_yield: bool = False,
) -> None:
    """Runs the training loop for the Autoencoder and GRBM.

    Args:
        title: The title for the training run.
        loss_fn: The loss function to use.
        solver: The D-Wave solver name.
        stop_grbm: The step at which to stop training the GRBM.
        num_reads: The number of reads for the QPU sampler.
        annealing_time: The annealing time for the QPU sampler.
        alpha: The learning rate for the GRBM.
        num_steps: The total number of training steps.
        device: The device used for training and sampling tensors.
        seed: Optional random seed for parameter initialization.
        use_srts: Whether to use the SRTS composite.
        use_automorphisms: Whether to use the Automorphism composite.
    """
    sampler, model, grbm = get_qpu_model_grbm(
        solver,
        device,
        m=m,
        t=t,
        use_srts=use_srts,
        use_automorphisms=use_automorphisms,
        allow_incomplete_yield=allow_incomplete_yield,
    )
    nprng = np.random.default_rng(seed)
    grbm.linear.data[:] = 0.1 * bit2spin_soft(
        torch.tensor(nprng.binomial(1, 0.5, grbm.n_nodes))
    )
    grbm.quadratic.data[:] = bit2spin_soft(
        torch.tensor(nprng.binomial(1, 0.5, grbm.n_edges))
    )

    model.train()
    grbm.train()

    # UNCOMMENT TO LOAD:
    # grbm.load_state_dict(torch.load("grbm.pt"))
    # model.load_state_dict(torch.load("model.pt"))

    opt_grbm = SGD(grbm.parameters(), lr=1e-3)
    opt_model = AdamW(model.parameters(), lr=1e-3)

    sample_params = dict(
        num_reads=num_reads,
        annealing_time=annealing_time,
        answer_mode="raw",
        auto_scale=False,
    )
    h_range, j_range = sampler.properties["h_range"], sampler.properties["j_range"]

    # Set up data
    train_loader, test_loader = get_dataset(num_reads)

    compute_mmd = MMDLoss().to(device)

    if os.path.isfile(f"{title}model.pt"):
        model.load_state_dict(torch.load(f"{title}model.pt"))
        warnings.warn('Trained model exists: try a different title')
        return
    if os.path.isfile(f"{title}grbm.pt"):
        grbm.load_state_dict(torch.load(f"{title}grbm.pt"))
        warnings.warn('Trained grbm exists: try a different title')
        return
    for step, (x, _) in enumerate(cycle(train_loader), 1):
        torch.cuda.empty_cache()
        if step > num_steps:
            break
        # Send data to device
        x = x.to(device).float()
        q = grbm.sample(
            sampler,
            prefactor=1,
            linear_range=h_range,
            quadratic_range=j_range,
            device=device,
            sample_params=sample_params,
        )

        # Train autoencoder
        stats = collect_stats(model, grbm, x, q, compute_mmd, compute_pkl)
        opt_model.zero_grad()
        (stats["bce"] + alpha * stats[loss_fn]).backward()
        # alpha ~ 1e-6
        opt_model.step()

        if step < stop_grbm:
            # NOTE: collecting stats again because the autoencoder has been updated.
            stats = collect_stats(model, grbm, x, q, compute_mmd, compute_pkl)
            opt_grbm.zero_grad()
            stats["quasi"].backward()
            opt_grbm.step()

        print(
            title,
            step,
            {
                k: f"{v.item():.4f}" if isinstance(v, torch.Tensor) else f"{v:.4f}"
                for k, v in stats.items()
            },
        )

        if step % 10 == 0:
            model.eval()

            xtest = next(iter(test_loader))[0].to(device)
            q = grbm.sample(
                sampler,
                prefactor=1,
                linear_range=h_range,
                quadratic_range=j_range,
                device=device,
                sample_params=sample_params,
            )
            save_viz(model, xtest, q, title=title)

            model.train()
            torch.save(grbm.state_dict(), f"{title}grbm.pt")
            torch.save(model.state_dict(), f"{title}model.pt")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--title",
        type=str,
        default="Default_",
        help="String used to prepend output files and print statements",
    )
    parser.add_argument(
        "--annealing_time",
        type=float,
        default=0.5,
        help="Annealing time in microseconds",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Learning rate for the GRBM",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=1000,
        help="Total number of training steps",
    )
    parser.add_argument(
        "--num_reads",
        type=int,
        default=1000,
        help="Number of reads for the QPU sampler at each step",
    )
    parser.add_argument(
        "--stop_grbm",
        type=int,
        default=500,
        help="Step at which to stop training the GRBM",
    )
    parser.add_argument(
        "--loss_fn",
        type=str,
        default="mmd",
        help="Loss function to use",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="Advantage2_system1.13",
        help="Leap QPU solver name",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=5,
        help="Parameter m for the Zephyr or dnx graph",
    )
    parser.add_argument(
        "--t",
        type=int,
        default=3,
        help="Parameter t for the Zephyr or dnx graph",
    )
    parser.add_argument(
        "--allow_incomplete_yield",
        action="store_true",
        help="Allow incomplete yield (default is False).",
    )
    parser.add_argument(
        "--use_srts",
        action="store_true",
        help="Use spin reversal transform (SRT) (default is False).",
    )
    parser.add_argument(
        "--use_automorphisms",
        action="store_true",
        help="Use automorphisms (default is False).",
    )
    args_ = parser.parse_args()

    args_dict = vars(args_)
    run(**args_dict)
