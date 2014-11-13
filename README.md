
# Step by Step Guide to Implementing a Multi-layer perceptron in C


###### Copyright

```c
/*
 * Copyright (c) 2004  John Bullinaria
 * Copyright (c) 2014  Javier Escalada Gómez (kerrigan29a@gmail.com)
 */
```

This article made by [Javier Escalada Gómez](http://www.github.com/Kerrigan29a)
is heavyly based on
[John Bullinaria's Step by Step Guide to Implementing a Neural Network in C](http://www.cs.bham.ac.uk/~jxb/NN/nn.html).

This document contains a step by step guide to implementing a neural network in
C. Obviously there are many types of neural network you could consider using -
here I shall concentrate on one particularly common and useful type, namely a
simple three-layer feed-forward back-propagation network (multi layer
perceptron).

This type of network will be useful when we have a set of input vectors and a
corresponding set of output vectors, and our system must produce an appropriate
output for each input it is given. Of course, if we already have a complete
noise-free set of input and output vectors, then a simple look-up table would
suffice. However, if we want the system to generalize, i.e. produce appropriate
outputs for inputs it has never seen before, then a neural network that has
learned how to map between the known inputs and outputs (i.e. our training set)
will often do a pretty good job for new inputs as well.

I shall assume that you are already familiar with C. For more details about
neural networks in general, you can use the
[comp.ai.neural-nets](http://groups.google.com/group/comp.ai.neural-nets)
newsgroup and the associated
[Neural Networks FAQ](ftp://ftp.sas.com/pub/neural/FAQ.html).

A single neuron (i.e. processing unit) takes it total input `In` and produces an
output activation `Out`. I shall take this to be the sigmoid function

```c
out = 1.0 / (1.0 + exp(-In));         /* out = sigmoid(In) */
```

though other activation functions are often used (e.g. linear or hyperbolic
tangent). This has the effect of squashing the infinite range of `In` into the
range 0 to 1. It also has the convenient property that its derivative takes the
particularly simple form

```c
sigmoid_derivative = sigmoid * (1.0 - sigmoid);
```

Typically, the input `In` into a given neuron will be the weighted sum of output
activations feeding in from a number of other neurons. It is convenient to think
of the activations flowing through layers of neurons. So, if there are
`NumUnits1` neurons in layer 1, the total activation flowing into our layer 2
neuron is just the sum over `Layer1Out[i] * Weight[i]`, where `Weight[i]` is the
strength/weight of the connection between unit `i` in layer 1 and our unit in
layer 2. Each neuron will also have a bias, or resting state, that is added to
the sum of inputs, and it is convenient to call this `Weight[0]`. We can then
write

```c
Layer2In = Weight[0];                   // start with the bias
for (i=1; i<=NumUnits1; i++) {
    Layer2In += Layer1Out[i] * Weight[i];
}
Layer2Out = 1.0 / (1.0 + exp(-Layer2In));
```

Normally layer 2 will have many units as well, so it is appropriate to write the
weights between unit `i` in layer 1 and unit `j` in layer 2 as an array
`Weight[i][j]`. Thus to get the output of unit `j` in layer 2 we have

```c
Layer2In[j] = Weight[0][j];
for (i=1; i<=NumUnits1; i++) {
    Layer2In[j] += Layer1Out[i] * Weight[i][j];
}
Layer2Out[j] = 1.0 / (1.0 + exp(-Layer2In[j]));
```

Remember that in C the array indices start from zero, not one, so we would
declare our variables as

```c
double Layer1Out[NumUnits1+1];
double Layer2In[NumUnits2+1];
double Layer2Out[NumUnits2+1];
double Weight[NumUnits1+1][NumUnits2+1];
```

(or, more likely, declare pointers and use `calloc` or `malloc` to allocate the
memory). Naturally, we need another loop to get all the layer 2 outputs

```c
for (j=1; j<=NumUnits2; j++ ) {
    Layer2In[j] = Weight[0][j];
    for (i=1; i<=NumUnits1; i++) {
        Layer2In[j] += Layer1Out[i] * Weight[i][j];
    }
    Layer2Out[j] = 1.0 / (1.0 + exp(-Layer2In[j]));
}
```

Three layer networks are necessary and sufficient for most purposes, so our
layer 2 outputs feed into a third layer in the same way as above

```c
for (j=1; j<=NumUnits2; j++) {      // j loop computes layer 2 activations
    Layer2In[j] = Weight12[0][j];
    for (i=1; i<=NumUnits1; i++) {
        Layer2In[j] += Layer1Out[i] * Weight12[i][j];
    }
    Layer2Out[j] = 1.0 / (1.0 + exp(-Layer2In[j]));
}
for (k=1; k<=NumUnits3; k++) {      // k loop computes layer 3 activations
    Layer3In[k] = Weight23[0][k];
    for (j=1; j<=NumUnits2; j++) {
        Layer3In[k] += Layer2Out[j] * Weight23[j][k];
    }
    Layer3Out[k] = 1.0 / (1.0 + exp(-Layer3In[k]));
}
```

The code can start to become confusing at this point - I find that keeping a
separate index `i`, `j`, `k` for each layer helps, as does an intuitive notation
for distinguishing between the different layers of weights `Weight12` and
`Weight23`. For obvious reasons, for three layer networks, it is traditional to
call layer 1 the Input layer, layer 2 the Hidden layer, and layer 3 the Output
layer. Our network thus takes on the familiar form that we shall use for the
rest of this document

<!--- TODO: Change img -->
![Structure of a generic neural network](nn.gif)

Also, to save getting all the In's and Out's confused, we can write `LayerNIn`
as `SumN`. Our code can thus be written


###### Compute hidden unit activations

```c
/* Compute hidden unit activations */
for (j=1; j<=NumHidden; j++ ) {
    SumH[p][j] = WeightIH[0][j];
    for( i=1; i<=NumInput; i++ ) {
        SumH[p][j] += Input[p][i] * WeightIH[i][j];
    }

    Hidden[p][j] = 1.0 / (1.0 + exp(-SumH[p][j]));
}
```


###### Compute output unit activations and errors

```c
/* Compute output unit activations and errors */
for(k=1; k<=NumOutput; k++) {
    SumO[p][k] = WeightHO[0][k];
    for( j = 1; j <= NumHidden; j++ ) {
        SumO[p][k] += Hidden[p][j] * WeightHO[j][k];
    }

    <<Compute sigmoidal output>>
    /* and */
    <<Compute sum squared error>>
    /* and */
    <<Compute delta for sigmoidal output and sum squared error>>

    /* or (forget about this at the moment of reading) */
    <<Compute linear output>>
    /* and */
    <<Compute delta for linear output and sum squared error>>

    /* or (forget about this at the moment of reading) */
    /* Compute sigmoidal output */
    /* and */
    <<Compute cross-entropy error>>
    /* and */
    <<Compute delta for sigmoidal output and cross-entropy error>>
}
```


###### Compute sigmoidal output

```c
/* Sigmoidal outputs */
Output[p][k] = 1.0 / (1.0 + exp(-SumO[p][k]));
```

Generally we will have a whole set of `NumPattern` training patterns, i.e. pairs
of vectors, `Input[p][i]` and `Target[p][k]` labelled by the index `p`.


###### Training data (XOR function)

```c
double  Input[NUM_TRAIN_INPUTS + 1][NUM_INPUT_UNITS + 1] = {
    { 0, 0, 0 },
    { 0, 0, 0 },
    { 0, 1, 0 },
    { 0, 0, 1 },
    { 0, 1, 1 }
};

double Target[NUM_TRAIN_INPUTS + 1][NUM_OUTPUT_UNITS + 1] = {
    { 0, 0 },
    { 0, 0 },
    { 0, 1 },
    { 0, 1 },
    { 0, 0 }
};
```

The network learns by minimizing some measure of the error of the network's
actual outputs compared with the target outputs. For example, the sum squared
error over all output units `k` and all training patterns `p` will be given by

```c
Error = 0.0;
for(p=1; p<=NumPattern; p++) {
    for(k=1; k<=NumOutput; k++) {
        Error += 0.5
            * (Target[p][k] - Output[p][k])
            * (Target[p][k] - Output[p][k]);
    }
}
```

(The factor of 0.5 is conventionally included to simplify the algebra in
deriving the learning algorithm.) If we insert the above code for computing the
network outputs into the `p` loop of this, we end up with


###### Compute sum squared error

```c
/* Sum squared error */
Error += 0.5 * (Target[p][k] - Output[p][k]) * (Target[p][k] - Output[p][k]);
```


###### Train with each pattern

```c
Error = 0.0;
/* Train with each pattern */
<<Unordered foreach>>

    <<Compute hidden unit activations>>

    /* Here we compute the error */
    <<Compute output unit activations and errors>>

    <<Back-propagate errors to hidden layer>>

    <<Update weights WeightIH>>

    <<Update weights WeightHO>>
}
```

The next stage is to iteratively adjust the weights to minimize the network's
error. A standard way to do this is by 'gradient descent' on the error function.
We can compute how much the error is changed by a small change in each weight
(i.e. compute the partial derivatives dError/dWeight) and shift the weights by a
small amount in the direction that reduces the error. The literature is full of
variations on this general approach - I shall begin with the 'standard on-line
back-propagation with momentum' algorithm. This is not the place to go through
all the mathematics, but for the above sum squared error we can compute and
apply one iteration (or `epoch`) of the required weight changes `DeltaWeightIH`
and `DeltaWeightHO` using


###### Compute delta for sigmoidal output and sum squared error

```c
/* Delta for sigmoidal outputs and sum squared error */
DeltaO[k] = (Target[p][k] - Output[p][k]) * Output[p][k] * (1.0 - Output[p][k]);
```


###### Back-propagate errors to hidden layer

```c
/* Back-propagate errors to hidden layer */
for(j=1; j<=NumHidden; j++) {
    SumDOW[j] = 0.0;
    for(k=1; k<=NumOutput; k++) {
        SumDOW[j] += WeightHO[j][k] * DeltaO[k];
    }

    DeltaH[j] = SumDOW[j] * Hidden[p][j] * (1.0 - Hidden[p][j]);
}
```


###### Update weights WeightIH

```c
/* Update weights WeightIH */
for(j=1; j<=NumHidden; j++) {
    DeltaWeightIH[0][j] = eta * DeltaH[j]
        + alpha * DeltaWeightIH[0][j];

    WeightIH[0][j] += DeltaWeightIH[0][j];

    for(i=1; i<=NumInput; i++) {
        DeltaWeightIH[i][j] = eta * Input[p][i] * DeltaH[j]
            + alpha * DeltaWeightIH[i][j];
        WeightIH[i][j] += DeltaWeightIH[i][j];
    }
}
```


###### Update weights WeightHO

```c
/* Update weights WeightHO */
for(k=1; k<=NumOutput; k++) {
    DeltaWeightHO[0][k] = eta * DeltaO[k]
        + alpha * DeltaWeightHO[0][k];

    WeightHO[0][k] += DeltaWeightHO[0][k];

    for(j=1; j<=NumHidden; j++) {
        DeltaWeightHO[j][k] = eta * Hidden[p][j] * DeltaO[k]
            + alpha * DeltaWeightHO[j][k];
        WeightHO[j][k] += DeltaWeightHO[j][k];
    }
}
```

(There is clearly plenty of scope for re-ordering, combining and simplifying the
loops here - I will leave that for you to do once you have understood what the
separate code sections are doing.) The weight changes `DeltaWeightIH` and
`DeltaWeightHO` are each made up of two components. First, the `eta` component
that is the gradient descent contribution. Second, the `alpha` component that is
a 'momentum' term which effectively keeps a moving average of the gradient
descent weight change contributions, and thus smoothes out the overall weight
changes. Fixing good values of the learning parameters `eta` and `alpha` is
usually a matter of trial and error. Certainly `alpha` must be in the range 0 to
1, and a non-zero value does usually speed up learning. Finding a good value for
`eta` will depend on the problem, and also on the value chosen for `alpha`. If
it is set too low, the training will be unnecessarily slow. Having it too large
will cause the weight changes to oscillate wildly, and can slow down or even
prevent learning altogether. (I generally start by trying eta = 0.1 and explore
the effects of repeatedly doubling or halving it.)


###### Eta and alpha values

```c
double eta = 0.5;
double alpha = 0.9;
```

The complete training process will consist of repeating the above weight updates
for a number of epochs until some error crierion is met, for example the `Error`
falls below some chosen small number. (Note that, with sigmoids on the outputs,
the `Error` can only reach exactly zero if the weights reach infinity! Note also
that sometimes the training can get stuck in a 'local minimum' of the error
function and never get anywhere the actual minimum.) So, we need to wrap the
last block of code in something like


###### Train

```c
/* Train */
for(epoch=0; epoch<MAX_EPOCH; epoch++) {

    <<Randomize the order of patterns>>

    <<Train with each pattern>>

    /* Show computation state each few epochs */
    if( epoch % 1000 == 0 ) {
        fprintf(stdout, "\rEpoch %-6d :   Error = %f", epoch, Error);
    }

    /* Stop learning when 'near enough' */
    if( Error < 0.000001 ) {
        fprintf(stdout, "\rEpoch %-6d :   Error = %f - DONE!\n", epoch, Error);
        goto print_result;
    }
}
```

If the training patterns are presented in the same systematic order during each
`epoch`, it is possible for weight oscillations to occur. It is therefore
generally a good idea to use a new random order for the training patterns for
each `epoch`. If we put the `NumPattern` training pattern indices `p` in random
order into an array `ranpat[]`, then it is simply a matter of replacing our
training pattern loop

```
for(p=1; p<=NumPattern; p++) {
```

with


###### Unordered foreach

```c
for(np=1; np<=NumPattern; np++) {
    p = ranpat[np];
```

Generating the random array `ranpat[]` is not quite so simple, but the following
code will do the job


###### Randomize the order of patterns

```c
/* Randomize the order of patterns */
for( p = 1; p <= NumPattern; p++ ) {
    ranpat[p] = p;
}
for( p = 1; p <= NumPattern; p++ ) {
    np = p + normalized_rand() * ( NumPattern + 1 - p );
    op = ranpat[p];
    ranpat[p] = ranpat[np];
    ranpat[np] = op;
}
```

Naturally, one must set some initial network weights to start the learning
process. Starting all the weights at zero is generally not a good idea, as that
is often a local minimum of the error function. It is normal to initialize all
the weights with small random values. `smallwt` is the maximum absolute size of
your initial weights.


###### Init the neural network with random values

```c
double smallwt = 0.5;

/* Seed random number function */
srand(time(NULL));

/* Initialize WeightIH and DeltaWeightIH */
for( j = 1; j <= NumHidden; j++ ) {
    for( i = 0; i <= NumInput; i++ ) {
        DeltaWeightIH[i][j] = 0.0;
        WeightIH[i][j] = ranged_rand(smallwt);
    }
}

/* Initialize WeightHO and DeltaWeightHO */
for( k = 1; k <= NumOutput; k ++ ) {
    for( j = 0; j <= NumHidden; j++ ) {
        DeltaWeightHO[j][k] = 0.0;
        WeightHO[j][k] = ranged_rand(smallwt);
    }
}
```


###### Ranged random number generator

```c
/*
 * This function generates values with the interval [-limit, limit)
 */
#define ranged_rand(limit) (2.0 * ( normalized_rand() - 0.5 ) * (limit))
```


###### Normalized random number generator

```c
/*
 * This is a good approximation to a uniform random distribution in the interval
 * [0, 1):
 * http://www.thinkage.ca/english/gcos/expl/c/lib/rand.html
 */
#define normalized_rand() ((double) rand() / (RAND_MAX + 1))
```

We now have enough code to put together a working neural network program. I have
cut and pasted the above code into the file [mlp.c](mlp.c).


###### mlp.c

```c
<<Copyright>>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define NUM_TRAIN_INPUTS    4

#define NUM_INPUT_UNITS     2
#define NUM_HIDDEN_UNITS    4
#define NUM_OUTPUT_UNITS    1

#define MAX_EPOCH           1000000



<<Normalized random number generator>>



<<Ranged random number generator>>



int main(void)
{
    int     i;
    int     j;
    int     k;
    int     p;
    int     np;
    int     op;
    int     ranpat[NUM_TRAIN_INPUTS + 1];
    int     epoch;

    int     NumPattern = NUM_TRAIN_INPUTS;
    int     NumInput = NUM_INPUT_UNITS;
    int     NumHidden = NUM_HIDDEN_UNITS;
    int     NumOutput = NUM_OUTPUT_UNITS;

    <<Training data (XOR function)>>

    double SumH[NUM_TRAIN_INPUTS + 1][NUM_HIDDEN_UNITS + 1];
    double WeightIH[NUM_INPUT_UNITS + 1][NUM_HIDDEN_UNITS + 1];
    double Hidden[NUM_TRAIN_INPUTS + 1][NUM_HIDDEN_UNITS + 1];
    double SumO[NUM_TRAIN_INPUTS + 1][NUM_OUTPUT_UNITS + 1];
    double WeightHO[NUM_HIDDEN_UNITS + 1][NUM_OUTPUT_UNITS + 1];
    double Output[NUM_TRAIN_INPUTS + 1][NUM_OUTPUT_UNITS + 1];
    double DeltaO[NUM_OUTPUT_UNITS + 1];
    double SumDOW[NUM_HIDDEN_UNITS + 1];
    double DeltaH[NUM_HIDDEN_UNITS + 1];
    double DeltaWeightIH[NUM_INPUT_UNITS + 1][NUM_HIDDEN_UNITS + 1];
    double DeltaWeightHO[NUM_HIDDEN_UNITS + 1][NUM_OUTPUT_UNITS + 1];
    double Error;

    <<Eta and alpha values>>

    <<Init the neural network with random values>>

    <<Train>>
    fprintf(stdout, "\rEpoch %-6d :   Error = %f - STOPPED!\n", epoch, Error);

print_result:
    /* Print network outputs */
    fprintf(stdout, "\nNETWORK DATA - EPOCH %d\n\nPat\t", epoch);
    for( i = 1; i <= NumInput; i++ ) {
        fprintf(stdout, "Input%-4d\t", i);
    }
    for( k = 1; k <= NumOutput; k++ ) {
        fprintf(stdout, "Expected%-4d\tReal%-4d\t", k, k);
    }
    for( p = 1; p <= NumPattern; p++ ) {
    fprintf(stdout, "\n%d\t", p);
        for( i = 1; i <= NumInput; i++ ) {
            fprintf(stdout, "%f\t", Input[p][i]);
        }
        for( k = 1; k <= NumOutput; k++ ) {
            fprintf(stdout, "%f\t%f\t", Target[p][k], Output[p][k]);
        }
    }
    fprintf(stdout, "\n\nDone!\n");
    return 0;
}
```

I've left plenty for the reader to do to convert this into a useful program, for
example:

* Read the training data from file
* Allow the parameters (`eta`, `alpha`, `smallwt`, `NumHidden`, etc.) to be
  varied during runtime
* Have appropriate array sizes determined and allocate them memory during
  runtime
* Saving of weights to file, and reading them back in again
* Plotting of errors, output activations, etc. during training

There are also numerous network variations that could be implemented, for
example:

* Batch learning, rather than on-line learning
* Alternative activation functions (linear, tanh, etc.)
* [Real (rather than binary) valued outputs require linear output functions](#appendix-i-real-values)
* [Cross-Entropy cost function rather than Sum Squared Error](#appendix-ii-cross-entropy-error)
* Separate training, validation and testing sets
* Weight decay / Regularization

But from here on, you're on your own. I hope you found this page useful...


# APPENDIX I: REAL VALUES


###### Compute linear output

```c
/* Linear outputs */
// Output[p][k] = SumO[p][k];
```

###### Compute delta for linear output and sum squared error

```c
/* Delta for linear outputs and sum squared error */
// DeltaO[k] = Target[p][k] - Output[p][k];
```


# APPENDIX II: CROSS-ENTROPY ERROR


###### Compute cross-entropy error

```c
/* Cross-entropy error */
// Error -= (Target[p][k] * log(Output[p][k]) + (1.0 - Target[p][k]) * log(1.0 - Output[p][k]));
```

###### Compute delta for sigmoidal output and cross-entropy error

```c
/* Delta for sigmoidal outputs and cross-entropy error */
// DeltaO[k] = Target[p][k] - Output[p][k];
```
