from mmd_ae import *
from dwave.preprocessing.composites import SpinReversalTransformComposite

solver = "Advantage2_system1.13"
qpu = DWaveSampler(solver=solver)
h_range, j_range = qpu.properties["h_range"], qpu.properties["j_range"]
G = subtile(zephyr_subgraph(qpu.to_networkx_graph(), 5), 3)
nodes = list(G.nodes)
edges = list(G.edges)
grbm = GRBM(nodes, edges)  # .to(device)


model = Autoencoder((1, 28, 28), grbm.n_nodes)  # .to(device)

model.load_state_dict(torch.load("model.pt"))
grbm.load_state_dict(torch.load("grbm.pt")
                     
# Print stats
num_reads = 4000
bs = 10000  # Full set is viable
train_loader, test_loader = get_dataset(bs=bs)
sample_params = dict(
    num_reads=num_reads, annealing_time=0.5, answer_mode="raw", auto_scale=False
)
q_orig = grbm.sample(
    qpu,
    prefactor=1,
    linear_range=h_range,
    quadratic_range=j_range,
    device=None,
    sample_params=sample_params,
)
num_srts = 40
assert num_reads%num_srts == 0
sample_params = dict(
    num_reads=num_reads//num_srts,
    annealing_time=0.5,
    answer_mode="raw",
    auto_scale=False,
    num_spin_reversal_transforms=num_srts,
)
q_srt = grbm.sample(
    SpinReversalTransformComposite(qpu),
    prefactor=1,
    linear_range=h_range,
    quadratic_range=j_range,
    device=None,
    sample_params=sample_params,
)

q_random = bit2spin_soft(torch.randint_like(q_orig, 2))

compute_mmd = MMDLoss(RadialBasisFunction())  # .to(device)


def compute_pkl(  # I will move this in the demo code, so it is at the top level of the module, and doesn't need to be repeated.
    grbm: GRBM,
    logits_data: torch.Tensor,
    spins_data: torch.Tensor,
    spins_model: torch.Tensor,
):
    probabilities = torch.sigmoid(logits_data)
    entropy = torch.nn.functional.binary_cross_entropy_with_logits(
        logits_data, probabilities
    )
    # bce = p(log(q)) + (1-p) log(1-q)
    cross_entropy = grbm.quasi_objective(spins_data, spins_model)
    pkl = cross_entropy - entropy
    return pkl


bqm_dist = {"trained": q_orig, f"SRTs{num_srts}": q_srt, "UID": q_random}
for n, q in bqm_dist.items():
    bs = 400
    rows = int(bs**0.5)
    xgen = model.decode(q[:bs]).sigmoid()
    xgengrid = make_grid(xgen, rows, pad_value=1)
    save_image(xgengrid, f"xgen_{n}.png")
    for xtest, _ in test_loader:
        stats = collect_stats(model, grbm, xtest, q, compute_mmd, compute_pkl)
        print(bqm_dist, stats)
        break  # 1 batch, nothing lost with break.
