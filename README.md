# Multi-layer perceptron

[custom_mlp.c](custom_mlp.c) is a custom multi-layes perceptron with a
3, 4, 4 structure in which each one of the 3 input nodes are connected with all
the hidden nodes but each hidden node is only connected with one output. The 4
weighted synapses between hidden nodes and output nodes are truncated to
`[-5, 5]`. The learning rates are diffeeren for each layer of synapses
(0.7 for input-hidden synapses and 0.07 for hidden-output synapses).

[mlp.c](mlp.c) is a standard multi-layes perceptron with a 2, 1, 4 structure.
