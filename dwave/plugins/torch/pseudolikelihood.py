# Kevin's review NOTE (kept for proof of work): I reviewed the logic and correctness of code, but
#   haven't checked documentation matches.
# Review NOTE: Reviewed 2026-09-16
"""Maximum pseudolikelihood estimation (MPLE) for graph-restricted Boltzmann machines (GRBMs).

Fits the linear and quadratic biases of a fully visible
:class:`dwave.plugins.torch.models.GraphRestrictedBoltzmannMachine` to observed spin data by
maximizing Besag's pseudolikelihood — the product, over sites, of each spin's conditional
probability given the rest of its configuration (J. Besag, *Statistical Analysis of Non-Lattice
Data*, The Statistician 24(3), 1975).

Model and objective. The GRBM defines the Boltzmann distribution ``P(s) ∝ exp(-H(s))`` over spin
configurations ``s ∈ {-1, +1}^N`` with Hamiltonian (energy function)

.. code-block:: text

    H(s) = sum_i h_i s_i + sum_{(i,j) in E} J_ij s_i s_j        # = grbm.forward(s)

Each site's conditional distribution is logistic in its local field
``theta_i(s) = h_i + sum_{j : (i,j) in E} J_ij s_j``:

.. code-block:: text

    P(s_i | s_rest) = sigmoid(-2 s_i theta_i(s))

so the negative log-pseudolikelihood (NLPL) of a dataset ``D`` — averaged over samples, summed
over sites — is

.. code-block:: text

    NLPL(h, J) = (1/|D|) sum_{s in D} sum_i softplus(2 s_i theta_i(s))

Why MPLE: the NLPL is jointly convex in ``(h, J)`` and involves no partition function and no
model sampling — unlike maximum-likelihood estimation (MLE), whose gradient requires equilibrium
samples from the model. For fully visible Boltzmann machines MPLE is a consistent estimator (it
recovers the true parameters as the sample size grows, at some statistical-efficiency cost
relative to MLE; see ``notes/qubie/pseudolikelihood_sample_complexity.md``).

Two fits share the objective:

* :func:`fit_pseudolikelihood` minimizes the NLPL over every bias ``(h, J)`` with L-BFGS (the
  limited-memory Broyden–Fletcher–Goldfarb–Shanno quasi-Newton method) in float64 working
  precision and writes the estimate back into the model's parameters.
* :func:`fit_inverse_temperature` holds the model's biases fixed as a reference Hamiltonian
  ``H*`` and minimizes the NLPL over the single scale parameter ``beta`` of the one-parameter
  family ``P_beta(s) ∝ exp(-beta * H*(s))`` — the inverse temperature, in physics vocabulary —
  by Newton's method on the NLPL derivative (SciPy's ``root_scalar``); it leaves the model
  untouched and returns ``beta`` with a standard error.

The two compose: on data drawn at inverse temperature ``beta`` from ``H*``, the joint fit
recovers ``beta * H*`` while the one-parameter fit against ``H*`` recovers ``beta``, and against
the joint fit's own estimate it returns exactly 1 (the joint optimum is stationary along its own
ray).

Scope notes:

* Fully visible models only — a GRBM with hidden nodes raises :class:`ValueError` (the
  pseudolikelihood of partially observed configurations would require marginalizing the hidden
  spins, which this module does not implement).
* Data columns follow ``grbm.nodes`` order, the plugin-wide convention.
* If the data were drawn at inverse temperature ``beta`` from a target Hamiltonian ``H*``, the
  joint MPLE recovers ``beta * H*``; rescale the fitted biases by ``1 / beta`` if ``H*`` itself
  is wanted, or — when ``H*`` is known — estimate ``beta`` alone with
  :func:`fit_inverse_temperature`.
* Degenerate (linearly separable) data — a site constant across all samples, or an edge whose
  endpoint product is constant — makes the MPLE diverge along that coordinate, exactly like
  logistic regression on separable data. Two routes:

  - **Default (``l2 = 0``)**: the fit **refuses such data** (raises :class:`ValueError`);
    remove the pinned statistics from the data/graph first (a lossless compression, e.g.
    prune constant sites, merge perfectly correlated pairs).
  - **Ridge / lasso (``l2 > 0`` and/or ``l1 > 0``)**: minimize
    NLPL + ``l1·(‖h‖₁ + ‖J‖₁)`` + ``l2·(‖h‖² + ‖J‖²)`` — the maximum a posteriori
    (MAP) estimate under a Laplace and/or Gaussian prior (elastic net when combined).
    The ``l1`` term is non-smooth, so L-BFGS treats it by subgradient — a practical
    heuristic, not a proximal method (no exact zeros). Either penalty bounds the optimum
    for **any** data, so separation (and parameter aliasing, e.g. an edge into a frozen
    site) stops being a special case; with ``l2 > 0`` the objective is strictly convex,
    making the optimum unique as well. The penalties ride on the *mean-per-sample* NLPL,
    so ``l2 ≈ 1/num_samples`` corresponds to a fixed unit prior that washes out as the
    sample grows (consistency preserved); every coordinate shrinks slightly, and
    ``l1 = l2 = 0`` remains the pure MPLE. (Ridge was removed 2026-08-03 in favor of prune/merge and
    reinstated 2026-08-21 by Kevin's explicit decision; see
    ``notes/qubie/pseudolikelihood_separable_data_options.md`` for the design discussion.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from dwave.plugins.torch.models import GraphRestrictedBoltzmannMachine
from scipy.optimize import root_scalar
from torch.nn import functional as F

from nimbus.spinbit import require_spin

__all__ = [
    "InverseTemperatureFit",
    "PseudolikelihoodFit",
    "fit_inverse_temperature",
    "fit_pseudolikelihood",
    "local_fields",
    "negative_log_pseudolikelihood",
]


def _validate_spins(spins: torch.Tensor, grbm: GraphRestrictedBoltzmannMachine) -> None:
    """Raise :class:`ValueError` unless ``spins`` is a valid (num_samples, n_nodes) spin batch."""
    # Kevin's review NOTE (kept for proof of work): good checks; can't think of any other
    #     assumptions to check. For what it's worth, divergence checks take place later.*
    #     this _validate_spins function is doing good by staying in scope.
    #     * divergence to mean pinned spins or pinned interaction values causing MPLE to diverge.
    if grbm.hidden_nodes:
        raise ValueError(
            "Pseudolikelihood requires a fully visible model, but the GRBM has "
            f"{len(grbm.hidden_nodes)} hidden node(s). Marginalizing hidden spins is out of "
            "scope for this module."
        )
    if spins.ndim != 2:
        raise ValueError(f"Expected spins of shape (num_samples, n_nodes), got {spins.shape}")
    if spins.shape[1] != grbm.n_nodes:
        raise ValueError(
            f"spins has {spins.shape[1]} columns but the model has {grbm.n_nodes} nodes "
            "(columns must follow `grbm.nodes` order)"
        )
    if spins.shape[0] == 0:
        raise ValueError("Expected at least one sample, got an empty batch")
    require_spin(spins)


def _use_dense_fields(n_nodes: int, n_edges: int) -> bool:
    """Whether the local fields go through a dense ``(N, N)`` coupling matrix (see below).

    The edge scatter costs one sequential pass over every (sample, edge) pair — about 30 ns each
    on CPU — while the dense matmul costs ``N^2`` flops per sample at several GFLOP/s, so the
    matmul wins once ``N^2`` is within a few hundred times the edge count (a 144-node complete
    graph: ~100x faster) and loses on the QPU-sized sparse native graphs (thousands of nodes, a
    few couplers per node), where the scatter stays. Tiny graphs always take the dense path.
    """
    return n_nodes <= 256 or n_nodes * n_nodes <= 400 * n_edges


def _coupling_matrix(
    quadratic: torch.Tensor, edge_idx_i: torch.Tensor, edge_idx_j: torch.Tensor, n_nodes: int
) -> torch.Tensor:
    """The symmetric ``(N, N)`` coupling matrix ``W[i, j] = W[j, i] = J_ij`` (zero diagonal).

    Built with two non-accumulating ``index_put`` calls, so it stays differentiable in
    ``quadratic`` (each edge is written exactly once per triangle).
    """
    # Review NOTE: This is a symmetric matrix that should only be used in calculating
    #   effective / local fields. Not to be interpreted as the J in s'Js (this would double count)
    w = torch.zeros(n_nodes, n_nodes, dtype=quadratic.dtype, device=quadratic.device)
    w = w.index_put((edge_idx_i, edge_idx_j), quadratic)
    return w.index_put((edge_idx_j, edge_idx_i), quadratic)


def _local_fields(
    spins: torch.Tensor,
    linear: torch.Tensor,
    quadratic: torch.Tensor,
    edge_idx_i: torch.Tensor,
    edge_idx_j: torch.Tensor,
) -> torch.Tensor:
    """Local fields ``theta_i(s) = h_i + sum_{e ∋ i} J_e * s_other`` for a (b, N) spin batch.

    All floating tensors must share one dtype and device; each edge contributes to the field of
    both of its endpoints. Differentiable with respect to ``linear`` and ``quadratic``. Two
    equivalent evaluations, chosen by :func:`_use_dense_fields`: ``spins @ W`` with the dense
    coupling matrix, or two ``index_add`` scatters over the edges.
    """
    if quadratic.numel() and _use_dense_fields(spins.shape[1], quadratic.shape[0]):
        w = _coupling_matrix(quadratic, edge_idx_i, edge_idx_j, spins.shape[1])
        return spins @ w + linear
    # Kevin's review NOTE (kept for proof of work): why not linear.repeat(bs, 1)?
    #   on second thought, i like `return fields + linear` because it makes explicit the
    #   result is linear + neighbour-contributions, whereas linear.repeat is less explicit.
    #   no strong opinion.
    fields = torch.zeros_like(spins)
    # Kevin's review NOTE (kept for proof of work): this is so clean, i love it.
    #   instinctively i thought this would do some double-counting. gave it more thought and
    #   walked through the logic. this is great. im getting deprecated fr T_T
    if quadratic.numel():
        fields = fields.index_add(1, edge_idx_i, quadratic * spins[:, edge_idx_j])
        fields = fields.index_add(1, edge_idx_j, quadratic * spins[:, edge_idx_i])
    return fields + linear


def local_fields(spins: torch.Tensor, grbm: GraphRestrictedBoltzmannMachine) -> torch.Tensor:
    """Compute every site's local field ``theta_i(s)`` under the model's current parameters.

    The local field of site ``i`` in configuration ``s`` is
    ``theta_i(s) = h_i + sum_{j : (i,j) in E} J_ij s_j``, so that the conditional distribution of
    the spin is ``P(s_i | s_rest) = sigmoid(-2 s_i theta_i(s))``.

    Args:
        spins (torch.Tensor): Spin data of shape (num_samples, n_nodes) with entries ±1, columns
            in ``grbm.nodes`` order.
        grbm (GraphRestrictedBoltzmannMachine): A fully visible model.

    Returns:
        torch.Tensor: Local fields of shape (num_samples, n_nodes), in the model's parameter
        dtype, differentiable with respect to the model's parameters.
    """
    _validate_spins(spins, grbm)
    s = spins.to(device=grbm.linear.device, dtype=grbm.linear.dtype)
    return _local_fields(s, grbm.linear, grbm.quadratic, grbm.edge_idx_i, grbm.edge_idx_j)


def negative_log_pseudolikelihood(
    spins: torch.Tensor,
    grbm: GraphRestrictedBoltzmannMachine,
    *,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    """Negative log-pseudolikelihood (NLPL) of spin data under the model's current parameters.

    A configuration's NLPL is the sum over sites of the negative log-probability of each spin
    conditioned on all others, ``sum_i softplus(2 s_i theta_i(s))`` — see the module docstring.
    The result is differentiable with respect to the model's parameters, so it can serve directly
    as a training loss or as a held-out evaluation metric (in nats; lower is better).

    Note: this evaluates in one pass and materializes (num_samples, n_edges) intermediates; chunk
    the batch externally if that does not fit in memory (:func:`fit_pseudolikelihood` chunks
    internally via ``sample_chunk``).

    Args:
        spins (torch.Tensor): Spin data of shape (num_samples, n_nodes) with entries ±1, columns
            in ``grbm.nodes`` order.
        grbm (GraphRestrictedBoltzmannMachine): A fully visible model.
        reduction (Literal["mean", "sum", "none"]): Reduction over samples — the mean (default) or
            sum of per-sample NLPLs, or ``"none"`` for the (num_samples,) vector.

    Returns:
        torch.Tensor: A scalar for ``"mean"``/``"sum"``, or a (num_samples,) tensor for
        ``"none"``, in the model's parameter dtype.

    Raises:
        ValueError: If the model has hidden nodes, the data is not a 2-D ±1 tensor of width
            ``grbm.n_nodes``, the batch is empty, or ``reduction`` is not recognized.
    """
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f'reduction must be "mean", "sum", or "none", got {reduction!r}')
    _validate_spins(spins, grbm)
    s = spins.to(device=grbm.linear.device, dtype=grbm.linear.dtype)
    theta = _local_fields(s, grbm.linear, grbm.quadratic, grbm.edge_idx_i, grbm.edge_idx_j)
    # Kevin's review NOTE (kept for proof of work): these three lines above are identical to the
    #   public `local_fields` function, but it's repeated here because `s` needs to be cast. the
    #   `local_fields` function only returns theta, so you'd have to cast `s` again anyways.
    per_sample = F.softplus(2.0 * s * theta).sum(dim=1)
    if reduction == "mean":
        return per_sample.mean()
    if reduction == "sum":
        return per_sample.sum()
    return per_sample


@dataclass(frozen=True)
class PseudolikelihoodFit:
    """Diagnostics returned by :func:`fit_pseudolikelihood`.

    Attributes:
        nlpl: Mean per-sample negative log-pseudolikelihood (NLPL) at the fitted parameters, in
            nats.
        grad_inf_norm: Infinity norm of the NLPL gradient at the solution — the first-order
            optimality measure.
        n_iterations: L-BFGS iterations performed (0 if the initial point already satisfied
            ``tolerance_grad``).
        n_function_evals: Objective/gradient evaluations performed inside the L-BFGS step (line
            searches evaluate more than once per iteration).
        converged: Whether ``grad_inf_norm <= tolerance_grad``. ``False`` typically means the
            iteration cap was hit (degenerate/separable data — the one case whose optimum sits at
            infinity — is rejected upfront with a :class:`ValueError` instead; see the module
            docstring).
    """

    nlpl: float
    grad_inf_norm: float
    n_iterations: int
    n_function_evals: int
    converged: bool


def _raise_if_separable(
    spins: torch.Tensor,
    edge_idx_i: torch.Tensor,
    edge_idx_j: torch.Tensor,
    n_edges: int,
    sample_chunk: int,
    device: torch.device,
    work_dtype: torch.dtype,
) -> None:
    """Raise :class:`ValueError` when the MPLE provably diverges along some coordinate.

    A site whose spin is constant across all samples (its empirical mean is ±1) or an edge whose
    endpoint product is constant sends the corresponding bias to ±infinity, exactly as logistic
    regression diverges on linearly separable data. The check is exact — ±1 values are summed in
    the float working dtype, where integer sums are exact (up to 2**24 samples even in float32) —
    so ``|sum| == num_samples`` iff the column is constant.
    """
    num_samples, n_nodes = spins.shape
    node_spin_sums = torch.zeros(n_nodes, dtype=work_dtype, device=device)
    edge_product_sums = torch.zeros(n_edges, dtype=work_dtype, device=device)
    # Kevin's review NOTE (kept for proof of work): `for lo in range...` here `lo` stands for lower?
    #   bad variable name. what's `sc`? -> refactored `lo` -> `chunk_start` and `sc` -> `spin_chunk`
    # Kevin's review NOTE (kept for proof of work): CHECKPOINT, START REVIEW HERE
    for chunk_start in range(0, num_samples, sample_chunk):
        spin_chunk = spins[chunk_start : chunk_start + sample_chunk].to(
            device=device, dtype=work_dtype
        )
        node_spin_sums += spin_chunk.sum(dim=0)
        if n_edges:
            edge_product_sums += (spin_chunk[:, edge_idx_i] * spin_chunk[:, edge_idx_j]).sum(dim=0)
    # Kevin's review NOTE (kept for proof of work): OK idk why I usually check with mean == +/- 1
    #   or std == 0, but counting (this implementation) is more reliable. silly me, AI wins again.
    #   imma get deprecated :c
    num_constant_nodes = int((node_spin_sums.abs() == num_samples).sum())
    num_constant_edges = int((edge_product_sums.abs() == num_samples).sum())
    # Kevin's review NOTE (kept for proof of work): Maybe it's better to raise error instead of
    #   warn. A simple lossless compression should be warranted if this is the case.
    #   (Addressed 2026-08-03: warn → raise; l2 escape hatch removed with the l2 option itself.)
    # Qubie: 2026-08-21: the l2 option is back per Kevin's decision; this check now runs only for
    #   the pure MPLE, l2 == 0, where divergence is real.
    if num_constant_nodes or num_constant_edges:
        raise ValueError(
            f"Data is degenerate: {num_constant_nodes} node(s) have a constant spin and "
            f"{num_constant_edges} edge(s) have a constant endpoint product across all "
            f"{num_samples} samples, so the maximum pseudolikelihood estimate diverges along "
            "those coordinates (as for separable logistic regression). Remove the pinned "
            "statistics from the data/graph first — a lossless compression (e.g. prune constant "
            "sites, merge perfectly correlated pairs) — and refit."
        )


def fit_pseudolikelihood(
    spins: torch.Tensor,
    grbm: GraphRestrictedBoltzmannMachine,
    *,
    max_iter: int = 500,
    tolerance_grad: float = 1e-7,
    tolerance_change: float = 0.0,
    sample_chunk: int | None = 2048,
    l1: float = 0.0,
    l2: float = 0.0,
) -> PseudolikelihoodFit:
    """Fit a GRBM to spin data by maximum pseudolikelihood, updating the model in place.

    Minimizes the mean-per-sample negative log-pseudolikelihood (NLPL; see the module docstring)
    over the model's linear and quadratic biases. The objective is convex, so the optimizer
    always starts from the origin (the optimum is initialization-independent) and is minimized
    with a full-batch L-BFGS (strong-Wolfe line search) on float64 working copies of the
    parameters; the solution is then written back into ``grbm.linear`` and ``grbm.quadratic``
    (cast to their existing dtype), so the initialization the model was constructed with is
    discarded. The working precision is always float64, so torch backends without float64
    support (e.g. Apple's Metal Performance Shaders, MPS) are not supported.

    Memory is bounded by processing the batch in ``sample_chunk``-row chunks (backpropagating each
    chunk before the next), so the peak intermediate is O(``sample_chunk`` × ``n_edges``) on the
    edge-scatter path, or O(``sample_chunk`` × ``n_nodes`` + ``n_nodes``²) on the dense
    coupling-matrix path taken for dense graphs (:func:`_use_dense_fields`), rather than
    O(``num_samples`` × ``n_edges``); the chunking is exact, not an approximation. The fit is
    deterministic given the data (no sampling anywhere).

    Args:
        spins (torch.Tensor): Spin data of shape (num_samples, n_nodes) with entries ±1, columns
            in ``grbm.nodes`` order.
        grbm (GraphRestrictedBoltzmannMachine): A fully visible model; its parameters are
            overwritten with the estimate.
        max_iter (int): Maximum L-BFGS iterations (a single ``LBFGS.step`` call iterates
            internally).
        tolerance_grad (float): First-order termination threshold on the gradient infinity norm.
        tolerance_change (float): L-BFGS stagnation threshold on parameter/loss change. The
            default 0.0 disables stagnation stops, so termination is by ``tolerance_grad`` or
            ``max_iter`` and the ``converged`` flag is a genuine first-order-optimality statement.
            A positive value (torch's own default is 1e-9) permits earlier stops that typically
            leave the gradient above ``tolerance_grad`` and hence report ``converged=False``.
        sample_chunk (int | None): Rows per chunk (default 2048); ``None`` processes the whole
            batch in one chunk.
        l1 (float): Laplace-prior (lasso) strength: adds ``l1·(‖h‖₁ + ‖J‖₁)`` to the
            objective. Like ``l2`` it bounds the optimum on separable data (the NLPL flattens
            along a separating direction while the penalty keeps growing), but the term is
            **non-smooth**: L-BFGS handles it by subgradient, so treat it as a practical
            heuristic — coefficients shrink but are not driven exactly to zero (no proximal
            step), the optimum need not be unique, and the first-order ``converged`` reading
            is approximate near kinks.
        l2 (float): Gaussian-prior (ridge) strength. ``0.0`` for both penalties (default) is
            the pure MPLE and refuses separable data; ``l2 > 0`` adds ``l2·(‖h‖² + ‖J‖²)`` —
            the MAP estimate, strictly convex and finite for any data (see the module
            docstring; ``l2 ≈ 1/num_samples`` corresponds to a fixed unit prior). The two
            penalties combine (elastic net). The reported ``nlpl`` stays the *unpenalized*
            NLPL at the solution; ``grad_inf_norm`` and ``converged`` describe the penalized
            objective actually minimized.

    Returns:
        PseudolikelihoodFit: Final NLPL, gradient norm, iteration/evaluation counts, and a
        convergence flag.

    Raises:
        ValueError: If the model has hidden nodes, the data is not a 2-D ±1 tensor of width
            ``grbm.n_nodes``, the batch is empty, an option is invalid, or — with
            ``l1 == l2 == 0`` only — the data is degenerate (a constant site or a constant
            edge-endpoint product, where the pure MPLE diverges; see
            :func:`_raise_if_separable`).
    """
    _validate_spins(spins, grbm)
    if sample_chunk is not None and sample_chunk < 1:
        raise ValueError(f"sample_chunk must be a positive int or None, got {sample_chunk}")
    for penalty_name, penalty_value in (("l1", l1), ("l2", l2)):
        if not (penalty_value >= 0.0) or penalty_value == float("inf"):
            raise ValueError(f"{penalty_name} must be a finite float >= 0, got {penalty_value}")

    device = grbm.linear.device
    # Kevin's review NOTE (kept for proof of work): idk what "MPS" device is. had to look it up.
    #     it stands for "Metal Performance Shaders". quoting pytorch below
    #     > MPS device enables high-performance training on GPU for macOS devices ...
    # Kevin's review NOTE (kept for proof of work): I removed "mps" device check because I
    #     don't care for no iCaveats LOL.
    work_dtype = torch.float64
    num_samples = spins.shape[0]
    n_edges = grbm.quadratic.shape[0]
    # Kevin's review NOTE (kept for proof of work): the memory cost of L-BFGS scales linearly in
    #     dimension*sample_size, so I think it's safe to use a larger default sample_chunk or even
    #     use a available-memory-based default/parameterization.
    chunk = num_samples if sample_chunk is None else sample_chunk
    edge_idx_i = grbm.edge_idx_i
    edge_idx_j = grbm.edge_idx_j

    if l1 == 0.0 and l2 == 0.0:
        # Pure MPLE only: with a penalty the optimum is finite for any data.
        _raise_if_separable(spins, edge_idx_i, edge_idx_j, n_edges, chunk, device, work_dtype)

    # Kevin's review NOTE (kept for proof of work): do we really need "init"? the objective is
    #   convex and MPLE with L-BFGS is super linear in convergence.
    #   (Addressed 2026-08-03: `init` removed — the fit always starts from the origin.)
    # Kevin's review NOTE (kept for proof of work): this feels a little unnecessary; just set both
    #   to require grad and that's it.
    #   (Addressed 2026-08-03: with the `init="model"` construction chain gone, requires_grad can
    #   be set at creation. The old two-step existed because `.detach().clone().to(dtype)` must
    #   run before `requires_grad_()` for the tensor to remain an autograd LEAF — see the review
    #   discussion in the prompt history.)
    h = torch.zeros(grbm.linear.shape[0], device=device, dtype=work_dtype, requires_grad=True)
    j = torch.zeros(n_edges, device=device, dtype=work_dtype, requires_grad=n_edges > 0)
    params = [h] + ([j] if n_edges else [])

    # Kevin's review NOTE (kept for proof of work): should this simply take in **kwargs_lbfgs?
    opt = torch.optim.LBFGS(
        params,
        max_iter=max_iter,
        tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change,
        line_search_fn="strong_wolfe",
    )

    # Kevin's review NOTE (kept for proof of work): do we really want NLPL? are there theoretical
    #   guarantees that it will converge to some quantity suitable for model selection?
    last_nlpl = 0.0

    def closure() -> torch.Tensor:
        # The objective is a mean over samples, so per-chunk backward calls accumulate the exact
        # gradient while each (chunk, n_edges) intermediate is freed before the next chunk.
        nonlocal last_nlpl
        opt.zero_grad()
        nlpl = torch.zeros((), device=device, dtype=work_dtype)
        for lo in range(0, num_samples, chunk):
            sc = spins[lo : lo + chunk].to(device=device, dtype=work_dtype)
            theta = _local_fields(sc, h, j, edge_idx_i, edge_idx_j)
            # Kevin's review NOTE (kept for proof of work):
            #   p(s | nbr) = exp(-s theta) / (exp(-theta) + exp(theta))
            #              = 1             / (exp(-theta(1 - s)) + exp(theta(1 + s)))
            #              = 1             / (exp(-theta(1 - s)) + exp(theta(1 + s)))
            #   if s = 1
            #              = 1             /  (1 + exp(2  theta))
            #   if s = -1
            #              = 1             / (exp(-2 theta) + 1 )
            #   so either way,
            #              = 1             /  (1 + exp(s 2 theta))
            #   - log(p(s=1 | nbr)) = log1 + log(1 + exp(s 2 theta))
            #                       =      + log(1 + exp(s 2 theta))
            #                       = softplus(s 2 theta)
            chunk_nlpl = F.softplus(2.0 * sc * theta).sum() / num_samples
            chunk_nlpl.backward()
            nlpl = nlpl + chunk_nlpl.detach()
        last_nlpl = float(nlpl)
        # Kevin's review NOTE (kept for proof of work): I think we can totally remove
        #   the l2 penalty. we're never really gonna work with it anyways. if statistics
        #   are pinned, we should remove them by lossless compression.
        # Qubie NOTE: (Addressed 2026-08-03: `l2` removed; the objective is the pure mean NLPL,
        #   and pinned statistics are rejected upfront by _raise_if_separable.)
        # Qubie NOTE: (Reinstated 2026-08-21 per Kevin's explicit decision — prompt 71 in
        #   exp/c2st_gbdt_benchmark/prompts_log.md: opt-in `l2` MAP/ridge; `last_nlpl` stays the
        #   pure NLPL, the returned objective and its gradient carry the penalty.)
        objective = nlpl
        if l1 or l2:
            penalty = torch.zeros((), device=device, dtype=work_dtype)
            if l1:
                penalty = penalty + l1 * (h.abs().sum() + (j.abs().sum() if n_edges else 0.0))
            if l2:
                penalty = penalty + l2 * (h.square().sum() + (j.square().sum() if n_edges else 0.0))
            penalty.backward()
            objective = objective + penalty.detach()
        return objective

    opt.step(closure)

    # One more evaluation at the solution for honest reporting: the final gradient and
    # the final pure NLPL. Called outside `step`, so it does not inflate the optimizer's counters.
    closure()
    # Kevin's review NOTE (kept for proof of work): this last closure call is for collecting
    #   statistics of the fit. might be worth removing. that should be a separate diagnostic, no?
    #   I guess it'd be tedious to redefine `closure`.
    grads = [p.grad.reshape(-1) for p in params if p.grad is not None]
    grad_inf_norm = float(torch.cat(grads).abs().max()) if grads else 0.0
    lbfgs_state = opt.state_dict()["state"].get(0, {})

    with torch.no_grad():
        grbm.linear.copy_(h.to(dtype=grbm.linear.dtype))
        grbm.quadratic.copy_(j.to(dtype=grbm.quadratic.dtype))

    return PseudolikelihoodFit(
        nlpl=last_nlpl,
        grad_inf_norm=grad_inf_norm,
        n_iterations=int(lbfgs_state.get("n_iter", 0)),
        n_function_evals=int(lbfgs_state.get("func_evals", 0)),
        converged=grad_inf_norm <= tolerance_grad,
    )


# Review NOTE: LGTM
@dataclass(frozen=True)
class InverseTemperatureFit:
    """Result of :func:`fit_inverse_temperature`: the estimate, its standard error, a flag.

    Attributes:
        beta: The maximum pseudolikelihood estimate of the inverse temperature — the scale
            ``beta`` for which ``P(s) ∝ exp(-beta * H*(s))`` best explains the data, ``H*`` being
            the model's fixed Hamiltonian. ``inf`` when no sample carries a single-spin excitation
            (every spin sits in its lower-energy state given its neighbours — the zero-temperature
            limit) and ``-inf`` when every spin sits in its higher-energy state; see
            :func:`fit_inverse_temperature`.
        beta_stderr: Sandwich (Godambe) standard error of ``beta``, treating the samples as
            independent draws; ``nan`` for an infinite ``beta`` or a single sample.
        converged: Whether the estimate met ``tolerance`` (an exact infinite limit counts as
            converged); ``False`` means ``max_iter`` ran out.
    """

    beta: float
    beta_stderr: float
    converged: bool


def _excitation_energies(
    spins: torch.Tensor,
    grbm: GraphRestrictedBoltzmannMachine,
    sample_chunk: int,
    work_dtype: torch.dtype,
) -> torch.Tensor:
    """Single-spin excitation energies ``eps_i(s) = H(s) - H(s^(i)) = 2 s_i theta_i(s)``.

    ``s^(i)`` is ``s`` with spin ``i`` flipped, so ``eps_i(s) > 0`` means the flip would lower
    the energy (the spin is excited against its local field) and ``eps_i(s) < 0`` that the spin is
    relaxed. Evaluated under the model's current parameters without autograd, in
    ``sample_chunk``-row chunks; the result has the data's shape, in ``work_dtype`` on the model's
    device.
    """
    device = grbm.linear.device
    linear = grbm.linear.detach().to(work_dtype)
    quadratic = grbm.quadratic.detach().to(work_dtype)
    energies = torch.empty(spins.shape, dtype=work_dtype, device=device)
    with torch.no_grad():
        for chunk_start in range(0, spins.shape[0], sample_chunk):
            spin_chunk = spins[chunk_start : chunk_start + sample_chunk].to(
                device=device, dtype=work_dtype
            )
            theta = _local_fields(spin_chunk, linear, quadratic, grbm.edge_idx_i, grbm.edge_idx_j)
            energies[chunk_start : chunk_start + sample_chunk] = 2.0 * spin_chunk * theta
    return energies


def fit_inverse_temperature(
    spins: torch.Tensor,
    grbm: GraphRestrictedBoltzmannMachine,
    *,
    max_iter: int = 100,
    tolerance: float = 1e-10,
    sample_chunk: int | None = 2048,
) -> InverseTemperatureFit:
    """Estimate the inverse temperature at which spin data were drawn from the model's Hamiltonian.

    Holds the model's biases fixed as the reference Hamiltonian ``H*`` and fits the single scale
    parameter of the one-parameter exponential family ``P_beta(s) ∝ exp(-beta * H*(s))`` — the
    inverse temperature ``beta``, in physics vocabulary — by maximum pseudolikelihood: the mean
    per-sample negative log-pseudolikelihood (NLPL; module docstring) restricted to that family,

    .. code-block:: text

        NLPL(beta) = (1/|D|) sum_{s in D} sum_i softplus(beta * eps_i(s)),
        eps_i(s)   = H*(s) - H*(s with spin i flipped) = 2 s_i theta_i(s),

    is minimized over ``beta``. The ``eps_i(s)`` are the single-spin excitation energies under
    ``H*``: positive when flipping spin ``i`` would lower the energy (the spin is excited against
    its local field ``theta_i``), negative when the spin is relaxed. The NLPL is a sum of softplus
    terms of a linear function of ``beta``, hence convex — strictly so once any ``eps_i(s)`` is
    nonzero — so the estimate is unique and needs no initialization. This is the temperature
    estimator of Raymond, Yarkoni and Andriyash (*Global warming: Temperature estimation in
    annealers*, Frontiers in ICT 3, 2016) — a logistic regression of every spin on its own local
    field with one shared slope — and the torch-native counterpart of Ocean's
    :func:`dwave.system.temperatures.maximum_pseudolikelihood_temperature`, which the plugin's
    ``GraphRestrictedBoltzmannMachine.estimate_beta`` wraps: same objective and estimate, but
    torch-native (runs on the model's device, exact chunking) with a standard error alongside.
    Ocean's effective-field computation loops over samples in Python, so at QPU scale this fit is
    orders of magnitude faster (measured ratios in ``nimbus/README.md``).

    Interpretation. If the data are Boltzmann samples of ``H*`` at some inverse temperature, the
    estimate is consistent for it. If they are not — annealer reads, a Markov chain that has not
    equilibrated — ``beta`` measures the rate of single-spin excitations relative to ``H*``, an
    *effective* inverse temperature, which is the quantity that decides how a model trained on
    such samples must be rescaled before it is programmed. The model itself is never modified:
    scale its biases by ``beta`` yourself if the calibrated Hamiltonian is wanted. The sign
    matters — a negative estimate says the spins are anti-aligned with their fields, usually a
    sign-convention error upstream.

    Algorithm. The excitation energies do not depend on ``beta``, so one pass over the data
    builds them (memory: one float64 tensor of the data's shape, plus ``sample_chunk``-bounded
    intermediates) and every later step is an elementwise pass over that tensor. The derivative
    ``NLPL'(beta) = (1/|D|) sum eps * sigmoid(beta * eps)`` is increasing in ``beta``, and
    Newton's method — :func:`scipy.optimize.root_scalar` with ``method="newton"`` and the analytic
    curvature ``NLPL''`` — finds its zero from ``beta = 0``, where the first step is the moment
    estimate ``-2 sum eps / sum eps^2``. No bracket or bisection safeguard is needed: each term
    ``eps * sigmoid(beta * eps)`` has a second derivative in ``beta`` with the sign of ``-beta``,
    so ``NLPL'`` is convex on ``beta < 0`` and concave on ``beta > 0``, and Newton started at that
    inflection point approaches the root monotonically from one side — never overshooting — with
    quadratic convergence near the optimum. (Halley's method would save passes but can overshoot
    on this objective, forfeiting the guarantee.) Iteration stops once a Newton step is at most
    ``tolerance * (1 + |beta|)``; the curvature and per-sample scores for the standard error are
    then evaluated once more, at the returned estimate.

    Limits. The derivative crosses zero only if some ``eps_i(s)`` is positive and some negative.
    Data with no excitation at all — every spin in its lower-energy state given its neighbours,
    e.g. a repeated ground state — put the maximizer at ``beta = +inf`` (zero temperature), and
    data with no relaxed spin at ``-inf``. Unlike the joint fit, which refuses separable data,
    these limits are returned as the exact estimate (``converged=True``, ``beta_stderr=nan``):
    they are the complete, physically meaningful answer, and calibration loops must handle them
    because excitations are rare in low-temperature annealer reads. Data on which ``H*`` has no
    local field anywhere (``eps ≡ 0``, e.g. an all-zero Hamiltonian) leave ``beta``
    unidentifiable and raise :class:`ValueError`.

    Standard error. The pseudolikelihood is a composite likelihood, so the estimate's sampling
    variance takes the sandwich (Godambe) form ``Var(u) / (|D| * NLPL''(beta)^2)``, with
    ``u_s = sum_i eps_i(s) sigmoid(beta * eps_i(s))`` the per-sample score (mean zero at the
    optimum) and ``NLPL''`` the mean curvature. It treats the samples as independent draws; for an
    autocorrelated Markov chain, thin the chain or scale by its effective sample size.

    Args:
        spins (torch.Tensor): Spin data of shape (num_samples, n_nodes) with entries ±1, columns
            in ``grbm.nodes`` order.
        grbm (GraphRestrictedBoltzmannMachine): A fully visible model whose current biases are
            the fixed reference Hamiltonian ``H*``; left unchanged.
        max_iter (int): Maximum Newton updates (default=100).
        tolerance (float): Positive relative stopping tolerance on ``beta`` (default=1e-10): the
            iteration stops once a Newton step is at most ``tolerance * (1 + |beta|)``.
        sample_chunk (int | None): Rows per chunk while building the excitation energies and
            evaluating the derivatives (default=2048); ``None`` processes the whole batch in one
            chunk.

    Returns:
        InverseTemperatureFit: The estimate, its standard error, and a convergence flag.

    Raises:
        ValueError: If the model has hidden nodes, the data is not a 2-D ±1 tensor of width
            ``grbm.n_nodes``, the batch is empty, an option is invalid, or the model's local
            fields vanish on every sample so that ``beta`` is unidentifiable.
    """
    _validate_spins(spins, grbm)
    if sample_chunk is not None and sample_chunk < 1:
        raise ValueError(f"sample_chunk must be a positive int or None, got {sample_chunk}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be a positive int, got {max_iter}")
    if not (tolerance > 0.0) or tolerance == math.inf:
        raise ValueError(f"tolerance must be a finite float > 0, got {tolerance}")

    work_dtype = torch.float64
    num_samples = spins.shape[0]
    chunk = num_samples if sample_chunk is None else sample_chunk
    # Review NOTE: Excitation energies is already 2x the local field.
    energies = _excitation_energies(spins, grbm, chunk, work_dtype)
    num_excited = int((energies > 0).sum())
    num_relaxed = int((energies < 0).sum())
    if num_excited == 0 and num_relaxed == 0:
        raise ValueError(
            "The model's local fields vanish on every sample (every single-spin excitation energy "
            "is zero, e.g. an all-zero Hamiltonian), so the pseudolikelihood does not depend on "
            "beta and the inverse temperature is unidentifiable."
        )
    if num_excited == 0 or num_relaxed == 0:
        # NLPL'(beta) keeps one sign on the whole line, so the pseudolikelihood is maximized only
        # in the limit beta -> +inf (no excitation) or -inf (no relaxed spin).
        return InverseTemperatureFit(
            beta=math.inf if num_excited == 0 else -math.inf, beta_stderr=math.nan, converged=True
        )

    def derivatives(beta: float) -> tuple[float, float, torch.Tensor]:
        """``NLPL'(beta)``, ``NLPL''(beta)`` and the per-sample scores, which sum to ``NLPL'``."""
        # Review NOTE: LGTM
        grad = 0.0
        curvature = 0.0
        scores = torch.empty(num_samples, dtype=work_dtype, device=energies.device)
        for chunk_start in range(0, num_samples, chunk):
            eps = energies[chunk_start : chunk_start + chunk]
            z = beta * eps
            sig = torch.sigmoid(z)
            # Review NOTE: 1st derivative LGTM (= chunk_scores)
            #   crap name tbh, should just be derivative
            chunk_scores = (eps * sig).sum(dim=1)
            scores[chunk_start : chunk_start + chunk] = chunk_scores
            # Review NOTE: grad LGTM too
            grad += float(chunk_scores.sum())
            # sigmoid(z) * sigmoid(-z) is the logistic variance, accurate where 1 - sigmoid(z)
            # would cancel.
            # Review NOTE: 2nd derivative LGTM
            curvature += float((eps.square() * sig * torch.sigmoid(-z)).sum())
        # Review NOTE: summed in loop, correctly average by sample size
        return grad / num_samples, curvature / num_samples, scores

    # Review NOTE: LGTM
    def newton_derivatives(beta: float) -> tuple[float, float]:
        """The ``(f, fprime)`` pair for SciPy's Newton: ``NLPL'`` and ``NLPL''`` at ``beta``."""
        grad, curvature, _ = derivatives(beta)
        return grad, curvature

    result = root_scalar(
        newton_derivatives,
        x0=0.0,
        fprime=True,
        method="newton",
        xtol=tolerance,
        rtol=tolerance,
        maxiter=max_iter,
    )
    beta = float(result.root)
    _, curvature, scores = derivatives(beta)  # at the returned root, for the standard error
    if num_samples < 2:
        beta_stderr = math.nan
    elif curvature > 0.0:
        beta_stderr = math.sqrt(float(scores.var(correction=1)) / num_samples) / curvature
    else:
        beta_stderr = math.inf
    return InverseTemperatureFit(
        beta=beta, beta_stderr=beta_stderr, converged=bool(result.converged)
    )


if __name__ == "__main__":

    def demo():
        # Smoke demo (unseeded — runs differ in the last digits by design): recover a planted
        # Hamiltonian on the complete 4-node graph from exact Boltzmann samples (enumerate all 16
        # configurations, sample from softmax(-H)); then hold the planted Hamiltonian fixed and
        # recover the inverse temperature of a second draw taken at beta = 1.7.
        nodes = list(range(4))
        edges = [(a, b) for a in nodes for b in nodes if a < b]
        h_true = 0.5 * (2 * torch.rand(len(nodes)) - 1)
        j_true = 1.0 * (2 * torch.rand(len(edges)) - 1)
        truth = GraphRestrictedBoltzmannMachine(
            nodes,
            edges,
            linear={n: float(v) for n, v in zip(nodes, h_true, strict=True)},
            quadratic={e: float(v) for e, v in zip(edges, j_true, strict=True)},
        )

        ints = torch.arange(2 ** len(nodes))
        configs = ((ints.unsqueeze(1) >> torch.arange(len(nodes))) & 1).float() * 2 - 1
        with torch.no_grad():
            energies = truth(configs)
        data = configs[torch.multinomial(torch.softmax(-energies, dim=0), 50_000, replacement=True)]

        model = GraphRestrictedBoltzmannMachine(nodes, edges)
        result = fit_pseudolikelihood(data, model)
        h_mae = float((model.linear.detach() - h_true).abs().mean())
        j_mae = float((model.quadratic.detach() - j_true).abs().mean())
        print(f"fit: {result}")
        print(f"mean absolute error: linear {h_mae:.4f}, quadratic {j_mae:.4f}")

        beta_true = 1.7
        cold_probs = torch.softmax(-beta_true * energies, dim=0)
        cold = configs[torch.multinomial(cold_probs, 50_000, replacement=True)]
        temperature = fit_inverse_temperature(cold, truth)
        print(f"inverse temperature: {temperature}")
        print(
            f"planted beta {beta_true}, estimate {temperature.beta:.4f} ± {temperature.beta_stderr:.4f}"
        )

    demo()
