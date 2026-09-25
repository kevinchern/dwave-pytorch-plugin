.. image:: https://img.shields.io/pypi/v/dwave-pytorch-plugin.svg
    :target: https://pypi.org/project/dwave-pytorch-plugin

.. image:: https://img.shields.io/pypi/pyversions/dwave-pytorch-plugin.svg
    :target: https://pypi.org/project/dwave-pytorch-plugin

.. image:: https://circleci.com/gh/dwavesystems/dwave-pytorch-plugin.svg?style=svg
    :target: https://circleci.com/gh/dwavesystems/dwave-pytorch-plugin

.. image:: https://codecov.io/gh/dwavesystems/dwave-pytorch-plugin/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/dwavesystems/dwave-pytorch-plugin

====================
dwave-pytorch-plugin
====================

.. start_pytorch_plugin_about

``dwave-pytorch-plugin`` provides `PyTorch <https://pytorch.org>`_ modules and samplers for
training Boltzmann machines with D-Wave quantum computers.

Boltzmann machines are energy-based models whose training requires samples from the model
distribution, which are costly to obtain classically. This plugin lets a D-Wave quantum processing
unit (QPU), or any classical `dimod <https://github.com/dwavesystems/dimod>`_ sampler, provide
those samples inside a standard PyTorch training loop, so that Boltzmann machines can be trained
on their own or as priors and layers of larger neural networks.

The package provides the following components.

* **Models** (``dwave.plugins.torch.models``). ``GraphRestrictedBoltzmannMachine`` is an Ising
  model on the nodes and edges of a graph, such as the working graph of a QPU. Its quadratic
  biases are stored as a dense, masked matrix, so energies, learning statistics, and effective
  fields are dense matrix products that run efficiently on GPUs. The model supports hidden units,
  exact conditional expectations of hidden units that are not adjacent to each other, a
  quasi-objective whose gradient is the gradient of the negative log likelihood, and estimation of
  the effective inverse temperature of a set of samples. ``DiscreteVariationalAutoencoder``,
  together with the ``pseudo_kl_divergence_loss`` of ``dwave.plugins.torch.models.losses``,
  supports training autoencoders with a Boltzmann machine prior over discrete latent variables.

* **Samplers** (``dwave.plugins.torch.samplers``). Samplers are ``torch.nn.Module`` objects that
  hold the model they sample from, so moving or saving a sampler moves or saves the model and the
  sampler's own state together. ``BlockSampler`` performs block-Gibbs or block-Metropolis
  sampling with persistent Markov chains, ``BipartiteGibbsSampler`` specializes it to restricted
  Boltzmann machines, and ``DimodSampler`` wraps any dimod sampler, including the
  ``DWaveSampler`` of `dwave-system <https://github.com/dwavesystems/dwave-system>`_, scaling and
  clipping the Hamiltonian before it is submitted. All samplers support conditional sampling of
  partially observed spins.

* **Neural network modules** (``dwave.plugins.torch.nn``). An ``Ising`` layer takes the biases
  of an Ising model as inputs and returns expected statistics of samples drawn by a dimod
  sampler, with a backward pass approximated by sample covariances; ``SpinStatistic``
  classes define the statistics it returns. Also included are ``LinearBlock``, ``SkipLinear``,
  and ``Affine`` layers, a ``GaussianKernel``, a ``MaximumMeanDiscrepancyLoss``, and, in
  ``dwave.plugins.torch.nn.functional``, the functional form of that loss and soft
  conversions between bits and spins.

* **Utilities** (``dwave.plugins.torch.utils`` and ``dwave.plugins.torch.tensor``). Conversions
  between tensors and dimod's Ising dictionaries and sample sets, and random spin generation.

.. end_pytorch_plugin_about

Installation
============

``dwave-pytorch-plugin`` requires Python 3.10 or later and is tested on Linux, macOS, and Windows.

Install from PyPI:

.. code-block:: bash

    pip install dwave-pytorch-plugin

Or install from source:

.. code-block:: bash

    pip install -r requirements.txt
    pip install .

PyTorch is installed as a dependency. To use a GPU, install a build of PyTorch for your hardware
first by following the `PyTorch installation instructions <https://pytorch.org/get-started/locally/>`_.

Sampling with a QPU requires a `Leap <https://cloud.dwavesys.com/leap/>`_ account. Run
``dwave config create`` to store your API token, or see the
`D-Wave documentation <https://docs.dwavequantum.com/>`_ for other ways to configure access to
Leap's solvers.

Examples
========

Complete scripts are in the repository's
`examples directory <https://github.com/dwavesystems/dwave-pytorch-plugin/tree/main/examples>`_.

Training a Boltzmann machine
----------------------------

The following example fits a fully visible Boltzmann machine on a 4-by-4 grid to synthetic data
using a classical block-Gibbs sampler. The negative phase of each step is a sweep of the sampler's
persistent Markov chains (persistent contrastive divergence).

.. code-block:: python

    import networkx as nx
    import torch

    from dwave.plugins.torch.models import GraphRestrictedBoltzmannMachine
    from dwave.plugins.torch.samplers import BlockSampler

    # A Boltzmann machine on a 4x4 grid: one spin per node, one coupling per edge
    graph = nx.grid_2d_graph(4, 4)
    grbm = GraphRestrictedBoltzmannMachine(graph.nodes, graph.edges)

    # Persistent block-Gibbs chains provide samples from the model (the negative phase)
    sampler = BlockSampler(grbm, num_chains=500, seed=0)

    # Training data: spins in {-1, +1} with one column per node. Here, all spins of a
    # configuration agree, so the model should learn ferromagnetic couplings.
    data = torch.sign(torch.randn(500, 1)).expand(-1, grbm.n_nodes)

    optimizer = torch.optim.SGD(grbm.parameters(), lr=0.05)
    for step in range(200):
        s_model = sampler.sample()  # advance every chain by one sweep
        optimizer.zero_grad()
        # The gradient of the quasi-objective is the gradient of the negative log likelihood
        grbm.quasi_objective(data, s_model).backward()
        optimizer.step()

    # Samples from the trained model reproduce the data: the spins (almost) always agree
    print(sampler.sample().mean(-1).abs().mean())  # close to 1

Sampling with a quantum computer
--------------------------------

To sample with a QPU, define the model on the QPU's working graph and wrap the ``DWaveSampler``
in a ``DimodSampler``. A QPU samples at an effective inverse temperature that differs from one,
so the Hamiltonian is scaled by a ``prefactor``, the reciprocal of an estimate of that inverse
temperature, and clipped to the programmable ranges before it is submitted. The training loop
is otherwise unchanged.

.. code-block:: python

    from dwave.system import DWaveSampler

    from dwave.plugins.torch.models import GraphRestrictedBoltzmannMachine
    from dwave.plugins.torch.samplers import DimodSampler

    qpu = DWaveSampler()
    graph = qpu.to_networkx_graph()
    grbm = GraphRestrictedBoltzmannMachine(graph.nodes, graph.edges)

    sampler = DimodSampler(
        grbm,
        qpu,
        prefactor=1 / 6.35,  # a ballpark estimate; refine it with grbm.estimate_beta
        linear_range=qpu.properties["h_range"],
        quadratic_range=qpu.properties["j_range"],
        # Return every read as its own sample and submit the Hamiltonian as is
        sample_kwargs=dict(num_reads=100, answer_mode="raw", auto_scale=False),
    )

    s_model = sampler.sample()  # a (100, n_nodes) tensor of spins
    # The inverse temperature at which the model was effectively sampled; ideally close to 1
    beta = grbm.estimate_beta(s_model)

Any dimod sampler can be wrapped in the same way; for example, a classical Markov chain Monte
Carlo sampler operating at unit inverse temperature needs no prefactor.

Hidden units
------------

Nodes passed as ``hidden_nodes`` are not observed in the data, and observations have one column
per visible node. The learning objective takes complete spin configurations, so the hidden units
of the data are filled in first: exactly, with their conditional expectations, when no two hidden
units are adjacent, or with conditional samples drawn by any sampler otherwise.

.. code-block:: python

    from dwave.plugins.torch.models import GraphRestrictedBoltzmannMachine
    from dwave.plugins.torch.samplers import BipartiteGibbsSampler

    # A restricted Boltzmann machine: 8 visible units fully connected to 4 hidden units
    visible = [f"v{i}" for i in range(8)]
    hidden = [f"h{j}" for j in range(4)]
    edges = [(v, h) for v in visible for h in hidden]
    grbm = GraphRestrictedBoltzmannMachine(visible + hidden, edges, hidden_nodes=hidden)
    sampler = BipartiteGibbsSampler(grbm, num_chains=100)

    x = ...  # observations of shape (batch_size, 8), one column per visible unit

    s_data = grbm.conditional_expectation(grbm.pad_visible(x))  # exact: hidden units are not adjacent
    s_data = sampler.complete(x)  # alternative: conditional samples, valid for any model
    grbm.quasi_objective(s_data, sampler.sample()).backward()

License
=======

Released under the Apache License 2.0. See the
`LICENSE file <https://github.com/dwavesystems/dwave-pytorch-plugin/blob/main/LICENSE>`_.

The use of the Boltzmann machine and discrete autoencoder implementations in this package with a
quantum computing system is protected by the intellectual property rights of D-Wave Quantum Inc.
and its affiliates. Their use with D-Wave's quantum computing systems requires access to D-Wave's
Leap quantum cloud service and is governed by the
`Leap Cloud Subscription Agreement <https://cloud.dwavesys.com/leap/legal/cloud_subscription_agreement/>`_.

Contributing
============

Ocean's `contributing guide <https://docs.ocean.dwavesys.com/en/stable/contributing.html>`_ has
guidelines for contributing to Ocean packages. See
`CONTRIBUTING.rst <https://github.com/dwavesystems/dwave-pytorch-plugin/blob/main/CONTRIBUTING.rst>`_
for how this package manages its release notes.

Testing
-------

Install the test requirements and run the unit tests from the root of the repository:

.. code-block:: bash

    pip install -r requirements.txt -r tests/requirements.txt
    python -m unittest
