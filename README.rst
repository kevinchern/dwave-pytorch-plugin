D-Wave PyTorch Plugin
=====================

This plugin provides an interface between D-Wave's quantum computers and
the PyTorch framework, including neural network modules for building
and training Boltzmann Machines along with various sampler utility functions.

Example
-------
Boltzmann Machines are probabilistic generative models for high-dimensional binary data.
The following example walks through a typical workflow for fitting Boltzmann Machines via maximum likelihood.

Define a Graph-Restricted Boltzmann Machine (GRBM) with a square graph. The quadratic biases are
stored in a dense ``(n_nodes, n_nodes)`` matrix whose entries outside the edges of the graph are
structurally zero, and all computations are dense matrix products.

.. code-block:: python

    import torch
    from torch.optim import SGD

    from dwave.plugins.torch.models import GraphRestrictedBoltzmannMachine as GRBM
    from dwave.plugins.torch.samplers import BlockSampler

    grbm = GRBM(nodes=["a", "b", "c", "d"], edges=[("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")])
    print("Linear biases:", grbm.linear)
    print("Quadratic biases:", grbm.quadratic)


Instantiate a `block-Gibbs sampler <https://en.wikipedia.org/wiki/Gibbs_sampling#Blocked_Gibbs_sampler>`_.
Nodes are partitioned into blocks of mutually non-adjacent nodes; here the partition is computed
automatically (pass ``colouring=lambda v: v in {"b", "d"}`` to choose the blocks yourself).
The sampler consists of three persistent Markov chains and performs ten sweeps at a constant unit
inverse temperature per call.

.. code-block:: python

    sampler = BlockSampler(grbm, num_chains=3, schedule=[1] * 10)


Create a batch of data and perform one likelihood-optimization step

.. code-block:: python

    x_data = torch.tensor([[1, -1, 1, -1], [-1, 1, 1, 1]], dtype=torch.float32)
    optimizer = SGD(grbm.parameters(), lr=1)
    x_model = sampler.sample()
    grbm.quasi_objective(x_data, x_model).backward()
    optimizer.step()
    print("Updated quadratic biases:", grbm.quadratic)

Samplers are ``torch.nn.Module`` objects that hold the model, so ``sampler.cuda()`` moves both the
model and the sampler's Markov chains to the GPU.

To use a `dimod <https://github.com/dwavesystems/dimod/>`_ sampler, replace the :code:`sampler = BlockSampler(...)` line with

.. code-block:: python

    from dwave.plugins.torch.samplers import DimodSampler
    from dwave.samplers import RandomSampler
    sampler = DimodSampler(grbm, RandomSampler(), sample_kwargs=dict(num_reads=5))


Hidden units
~~~~~~~~~~~~
Nodes listed in ``hidden_nodes`` are not observed in the data, which has one column per visible
node. When hidden units are not connected to each other, their conditional expectations given the
data are computed exactly:

.. code-block:: python

    from dwave.plugins.torch.samplers import BipartiteGibbsSampler

    rbm = GRBM(nodes=["v1", "v2", "h1", "h2"],
               edges=[("v1", "h1"), ("v1", "h2"), ("v2", "h1"), ("v2", "h2")],
               hidden_nodes=["h1", "h2"])
    sampler = BipartiteGibbsSampler(rbm, num_chains=3, schedule=[1] * 10)
    x_data = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32)
    rbm.quasi_objective(x_data, sampler.sample(), kind="exact-disc").backward()

Otherwise, sample the hidden units conditioned on the data with any sampler, for example
``rbm.quasi_objective(x_data, x_model, kind="sampling", sampler=sampler)``. Conditional sampling is
also available directly: ``sampler.sample(rbm.pad_visible(x_data))`` samples the ``torch.nan``
entries of its argument while keeping the observed spins fixed.


License
-------

Released under the Apache License 2.0. See LICENSE file.
