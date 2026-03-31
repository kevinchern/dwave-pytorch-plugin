from itertools import cycle
from collections.abc import Callable
from typing import TYPE_CHECKING, Any
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

import dimod
from dwave.plugins.torch.models.boltzmann_machine import (
    GraphRestrictedBoltzmannMachine as GRBM,
)
from dwave.plugins.torch.nn.functional import bit2spin_soft, spin2bit_soft
from dwave.system import DWaveSampler
from minorminer.subgraph import find_subgraph

from dwave.system.composites import FixedEmbeddingComposite
from dwave.preprocessing.composites import SpinReversalTransformComposite
from dwave.experimental.automorphism.automorphism_composite import AutomorphismComposite

if TYPE_CHECKING:
    import dimod


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

    def __init__(self, input_shape: tuple[int, int, int], n_bits: int) -> None:
        """Initialize the autoencoder model.

        Args:
            shape: shape of the input images, as (channels, height, width).
            n_bits: number of bits for the latent representation.
        """
        super().__init__()
        _, h, w = input_shape
        chidden = 1
        depth_fcnn = 3
        depth_cnn = 3
        dropout = 0.0
        self.encoder = nn.Sequential(
            ConvolutionNetwork([chidden] * depth_cnn, input_shape),
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


def save_gen(
    model: Autoencoder,
    title: str,
    q: torch.Tensor,
):
    """Saves generated images from the model.

    Args:
        model: The Autoencoder model used for decoding the samples.
        title: Prefix used for generated image files.
        q: Latent representation tensor (QPU samples).
    """
    rows = int(q.shape[0] ** 0.5)
    with torch.no_grad():
        # Save images
        xgen = model.decode(q).sigmoid()
        xgengrid = make_grid(xgen, rows, pad_value=1)
        save_image(xgengrid, f"{title}xgen.png")


def save_gen_multiple_methods(
    grbm: GRBM,
    qpu: DWaveSampler,
    emb: dict[Any, tuple[Any, ...]],
    model: Autoencoder,
    device: str | torch.device,
    sample_params: dict[str, Any] | None = None,
    seed: int | np.random.Generator | None = None,
    title: str = "",
    num_programming_transformations: int = 5,
    num_reads: int | None = None,
) -> None:
    """Generate and save samples using multiple sampler configurations.

    Iterates over combinations of SRTS and automorphism composites,
    sampling from the GRBM with each configuration and saving visualizations.

    Args:
        grbm: The graph restricted Boltzmann machine.
        qpu: The D-Wave sampler (QPU).
        emb: The embedding mapping from GRBM nodes to QPU qubits.
        model: The Autoencoder for decoding sampled latent representations.
        device: The device to run computations on.
        sample_params: Parameters for GRBM sampling. Defaults to None.
        seed: Random seed for reproducibility. Defaults to None.
        title: Prefix for output filenames. Defaults to empty string.
        num_programming_transformations: Number of programmings per SRT or automorphism. Defaults to 5.
    """
    if num_reads is None:
        num_reads = num_programming_transformations ** 4
    if sample_params is None:
        sample_params = dict(annealing_time=0.5, answer_mode="raw", auto_scale=False)
    else:
        sample_params = sample_params.copy()

    for use_srts in [False, True]:
        if use_srts:
            sample_params["num_spin_reversal_transforms"] = (
                num_programming_transformations
            )
            num_reads_per_srt = num_reads // num_programming_transformations
        else:
            num_reads_per_srt = num_reads

        for use_automorphisms in [False, True]:
            if use_automorphisms:
                sample_params["num_automorphisms"] = num_programming_transformations
                sample_params["num_reads"] = (
                    num_reads_per_srt // num_programming_transformations
                )
            else:
                sample_params["num_reads"] = num_reads_per_srt
            print("DEBUG statement", sample_params)
            sampler = get_sampler(
                qpu, emb, use_srts, use_automorphisms, grbm.edges, seed
            )

            q = grbm.sample(
                sampler,
                linear_range=qpu.properties["h_range"],
                quadratic_range=qpu.properties["j_range"],
                prefactor=1,
                device=device,
                sample_params=sample_params,
            )
            assert (
                len(q) == sample_params["num_reads"]
            ), f"Expected num_reads to be {sample_params['num_reads']} after adjusting for SRTs and automorphisms q.shape={q.shape} sample_params={sample_params0}"
            save_gen(
                model,
                f"{title}_S{use_srts}A{use_automorphisms}NPT{num_programming_transformations}",
                q,
            )


def save_viz(
    model: Autoencoder,
    x: torch.Tensor,
    q: torch.Tensor,
    title: str = "",
    max_samples: int = 400,
) -> None:
    """Saves visualizations of the input, generated, and reconstructed images.

    Args:
        model: The Autoencoder.
        x: Input tensor.
        q: Latent representation tensor.
        title: Prefix used for generated image files.
    """
    bs = min(x.shape[0], max_samples)
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
        to_coord = dnx.pegasus_coordinates(m=G.graph["rows"]).linear_to_pegasus
        co_index = 0
    elif G.graph["family"] == "chimera":
        to_coord = dnx.chimera_coordinates(
            m=G.graph["rows"], t=G.graph["tile"]
        ).linear_to_chimera
        co_index = 2
    else:
        raise ValueError("Unknown case")
    return {n: str(to_coord(n)[co_index]) for n in G.nodes}


def get_model_grbm_qpu_emb(
    solver: str,
    device: str,
    m: int = 5,
    t: int = 3,
    dnx_family: str = "zephyr",
    timeout: int = 60,
    allow_incomplete_yield: bool = False,
    orientation_hint: bool = True,
    input_shape: tuple[int, int, int] = (1, 28, 28),
    seed: int | None = None,
) -> tuple[Autoencoder, GRBM, DWaveSampler, dict[Any, tuple[Any, ...]]]:
    """Sets up the QPU, GRBM, and Autoencoder model.

    Args:
        solver: The D-Wave solver name.
        device: The device to run the model on, typically "cuda" or "cpu".
        m: Rows and columns of a small Chimera graph
        t: Tile parameter of a small Chimera graph
        dnx_family: The family of D-Wave hardware to target for the subgraph embedding. This is used to determine the structure of the Chimera graph to embed, which should be compatible with the target hardware. For example, "zephyr" would indicate that we want to embed a Zephyr subgraph, which is a specific type of Chimera graph with certain connectivity properties.
        timeout: timeout for chimera graph search.
        input_shape: The shape of the input images for the Autoencoder.
        seed: Seed for pseudo-random components (find_subgraph).
    Returns:
        A tuple containing the QPU sampler, Autoencoder model, and GRBM.
    """
    # Set up QPU and QPU parameters
    qpu = DWaveSampler(solver=solver)
    # Instantiate model
    T = qpu.to_networkx_graph()
    if dnx_family == "zephyr":
        S = dnx.zephyr_graph(m=m, t=t)
    elif dnx_family == "pegasus":
        S = dnx.pegasus_graph(m)
    elif dnx_family == "chimera":
        S = dnx.chimera_graph(m=m, n=m, t=t)
    else:
        raise ValueError(f"Unknown dnx_family: {dnx_family}")
    if orientation_hint:
        node_labels = (node_coloring(S), node_coloring(T))
    else:
        node_labels = None

    emb = find_subgraph(
        S, T, timeout=timeout, as_embedding=True, node_labels=node_labels, seed=seed
    )  # TO DO: add orientation hinting
    if len(emb) < S.number_of_nodes():
        if not allow_incomplete_yield:
            raise RuntimeError(
                f"Failed to find an embedding of the {dnx_family} graph "
                f"with m={m} and t={t} within the timeout {timeout}s."
                "Consider a simpler graph, smaller m and/or t, or larger timeout."
            )
        else:
            assert (
                dnx_family == "zephyr"
            ), "allow_incomplete_yield is currently only implemented for zephyr family graphs"
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
    model = Autoencoder(input_shape=input_shape, n_bits=grbm.n_nodes).to(device)
    return model, grbm, qpu, emb


def get_sampler(
    qpu: DWaveSampler,
    emb: dict[Any, tuple[Any, ...]],
    use_srts: bool,
    use_automorphisms: bool,
    edges: list[tuple[Any, Any]],
    seed: int | np.random.Generator | None = None,
) -> dimod.Sampler:
    sampler = FixedEmbeddingComposite(qpu, emb)
    if use_automorphisms:
        S = nx.Graph()
        S.add_nodes_from(emb.keys())
        S.add_edges_from(edges)
        sampler = AutomorphismComposite(sampler, G=S, seed=seed)
    if use_srts:
        sampler = SpinReversalTransformComposite(sampler, seed=seed)

    for key in ["h_range", "j_range"]:
        sampler.properties[key] = qpu.properties[key]  # type: ignore
    return sampler


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


def print_stage(title: str, step: int | None, stats: dict[str, torch.Tensor]) -> None:
    """Print stats for the current stage of training.
    Args:
        title: The title for the training run.
        step: The current training step.
        stats: A dictionary containing the statistics to be printed.
    """
    print(
        title,
        step,
        {
            k: f"{v.item():.4f}" if isinstance(v, torch.Tensor) else f"{v:.4f}"
            for k, v in stats.items()
        },
    )


def eval_stage(
    model: nn.Module,
    grbm: GRBM,
    test_loader: DataLoader,
    sampler: Any,
    sample_params: dict[str, Any],
    device: str | torch.device,
    title: str,
    post_training: bool = False,
    qpu: DWaveSampler | None = None,
    emb: dict | None = None,
    seed: int | np.random.Generator | None = None,
) -> None:
    """Evaluates the model and GRBM on the test set and saves result visualization.

    Args:
        model: The autoencoder model.
        grbm: The graph restricted Boltzmann machine.
        test_loader: DataLoader for the test set.
        sampler: The sampler used for the GRBM.
        sample_params: Parameters for sampling from the GRBM.
        device: The device to run the computations on.
        title: String appended to filenames for saved visualizations.
        post_training: Whether this evaluation is occurring after training has completed. If True, uses multiple sampler configurations for visualization.
    """
    model.eval()
    if post_training:
        save_gen_multiple_methods(
            grbm=grbm,
            qpu=qpu,
            emb=emb,
            model=model,
            device=device,
            sample_params=sample_params,
            seed=seed,
            title=title,
        )
    else:
        xtest = next(iter(test_loader))[0].to(device)
        q = grbm.sample(
            sampler,
            prefactor=1,
            linear_range=sampler.properties["h_range"],
            quadratic_range=sampler.properties["j_range"],
            device=device,
            sample_params=sample_params,
        )
        save_viz(model, xtest, q, title=title)
    model.train()


def train(
    model: nn.Module,
    grbm: GRBM,
    train_loader: DataLoader,
    num_steps: int,
    stop_grbm: int,
    sampler: Any,
    sample_params: dict[str, Any],
    device: str | torch.device,
    compute_mmd: nn.Module,
    reg_coefficient: float,
    loss_fn: str,
    title: str,
    opt_model: torch.optim.Optimizer,
    opt_grbm: torch.optim.Optimizer,
    test_loader: DataLoader,
    print_every: int | None,
    eval_every: int | None,
    save_every: int | None,
) -> dict[str, torch.Tensor] | None:
    """TODO: Add detailed documentation for the training loop and arguments."""
    stats = None

    for step, (x, _) in enumerate(cycle(train_loader), 1):
        torch.cuda.empty_cache()
        if step > num_steps:
            break
        # Send data to device
        x = x.to(device).float()
        q = grbm.sample(
            sampler,
            prefactor=1,
            linear_range=sampler.properties["h_range"],
            quadratic_range=sampler.properties["j_range"],
            device=device,
            sample_params=sample_params,
        )

        # Train autoencoder
        stats = collect_stats(model, grbm, x, q, compute_mmd, compute_pkl)
        opt_model.zero_grad()
        (stats["bce"] + reg_coefficient * stats[loss_fn]).backward()
        # reg_coefficient ~ 1e-6
        opt_model.step()

        if step < stop_grbm:
            # NOTE: collecting stats again because the autoencoder has been updated.
            stats = collect_stats(model, grbm, x, q, compute_mmd, compute_pkl)
            opt_grbm.zero_grad()
            stats["quasi"].backward()
            opt_grbm.step()
        if print_every and step % print_every == 0:
            print_stage(title, step, stats)

        if eval_every and step % eval_every == 0:
            eval_stage(
                model,
                grbm,
                test_loader,
                sampler,
                sample_params,
                device,
                title,
            )

        if save_every and step % save_every == 0:
            torch.save(grbm.state_dict(), f"{title}grbm.pt")
            torch.save(model.state_dict(), f"{title}model.pt")
    return stats


def run(
    *,
    title: str,
    loss_fn: str,
    solver: str,
    stop_grbm: int,
    num_reads: int,
    annealing_time: float,
    reg_coefficient: float,
    num_steps: int,
    device: str = "cuda",
    seed: int | None = None,
    m: int,
    t: int,
    use_srts: bool = False,
    use_automorphisms: bool = False,
    allow_incomplete_yield: bool = False,
    dnx_family: str = "zephyr",
    print_every: int | None = 10,
    eval_every: int | None = 50,
    save_every: int | None = 100,
) -> None:
    """Runs the training loop for the Autoencoder and GRBM.

    Args:
        title: The title for the training run.
        loss_fn: The loss function used to regularize bce.
        solver: The D-Wave solver name.
        stop_grbm: The step at which to stop training the GRBM.
        num_reads: The number of reads for the QPU sampler.
        annealing_time: The annealing time for the QPU sampler.
        reg_coefficient: MMD regularization strength.
        num_steps: The total number of training steps.
        device: The device used for training and sampling tensors.
        seed: Optional random seed for parameter initialization.
        m: Rows and columns of the target hardware subgraph.
        t: Tile parameter of the target hardware subgraph.
        allow_incomplete_yield: Whether to fall back to a yield-tolerant embedding.
        dnx_family: Hardware family to target ("zephyr", "pegasus", or "chimera").
        use_srts: Whether to use the SRTS composite.
        use_automorphisms: Whether to use the Automorphism composite.
        print_every: Log training stats every this many steps. None disables logging.
        eval_every: Run evaluation every this many steps. None disables evaluation.
        save_every: Save checkpoints every this many steps. None disables saving.
    """
    model, grbm, qpu, emb = get_model_grbm_qpu_emb(
        solver,
        device,
        m=m,
        t=t,
        allow_incomplete_yield=allow_incomplete_yield,
        dnx_family=dnx_family,
        seed=seed,
    )
    sampler = get_sampler(
        qpu,
        emb,
        use_srts=use_srts,
        use_automorphisms=use_automorphisms,
        edges=grbm.edges,
        seed=seed,  # Reuse of seed with find_subgraph is not a practical concern.
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

    # Set up data
    train_loader, test_loader = get_dataset(num_reads)

    compute_mmd = MMDLoss().to(device)
    train_model = True
    if os.path.isfile(f"{title}model.pt"):
        model.load_state_dict(torch.load(f"{title}model.pt"))
        print("Trained model exists: try a different title")
        train_model = False
    if os.path.isfile(f"{title}grbm.pt"):
        grbm.load_state_dict(torch.load(f"{title}grbm.pt"))
        print("Trained grbm exists: try a different title")
        train_model = False
    if train_model:
        train(
            model,
            grbm,
            train_loader,
            num_steps,
            stop_grbm,
            sampler,
            sample_params,
            device,
            compute_mmd,
            reg_coefficient,
            loss_fn,
            title,
            opt_model,
            opt_grbm,
            test_loader,
            print_every,
            eval_every,
            save_every,
        )
    else:
        print("Training skipped, files exist")
    if not train_model or eval_every is None:
        eval_stage(
            model,
            grbm,
            test_loader,
            sampler,
            sample_params,
            device,
            title,
            post_training=True,
            qpu=qpu,
            emb=emb,
            seed=seed,
        )
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
        "--reg_coefficient",
        type=float,
        default=1.0,
        help="MMD regularization strength",
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
        help="Loss function to use for training the autoencoder, either 'mmd' or 'pkl'",
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
        "--dnx_family",
        type=str,
        default="zephyr",
        help="zephyr, pegasus or chimera family (as target model)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility, None by default",
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
