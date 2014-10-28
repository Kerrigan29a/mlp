////////////////////////////////////////////////////////////////////////////
//MLP neural network in C++
// Original source code by Dr Phil Brierley
// www.philbrierley.com
// Translated to C++ - dspink Sep 2005
// This code may be freely used and modified at will
// C++ Compiled using Bloodshed Dev - C++ free compiler http://www.bloodshed.net /
//C Compiled using Pelles C free windows compiler http://smorgasbordet.com /
////////////////////////////////////////////////////////////////////////////

//
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>


////Data dependent settings ////
#define num_input_nodes  3
#define num_ouput_nodes  4


////User defineable settings ////
#define num_hidden_nodes 4
const int       num_epoch = 500;
const double    learning_factor_IH = 0.7;
const double    learning_factor_HO = 0.07;

////variables ////
int             num_train_inputs = 0;
double          desviation_error = 0.0;
double          obtained_output = 0.0;
double          root_mean_square_error = 0.0;

//the outputs of the hidden neurons
double          hidden_node_values[num_hidden_nodes];

//the weights
double          weights_IH[num_input_nodes][num_hidden_nodes];
double          weights_HO[num_hidden_nodes];

//the data
int             train_inputs[num_ouput_nodes][num_input_nodes];
int             train_output[num_ouput_nodes];



//***********************************
//calculates the network output
void calculate_error(void)
{
    // calculate the outputs of the hidden neurons
    // the hidden neurons are tanh
    int i;
    int j;
    for (i = 0; i < num_hidden_nodes; i++) {
        hidden_node_values[i] = 0.0;
        for (j = 0; j < num_input_nodes; j++) {
            hidden_node_values[i] += train_inputs[num_train_inputs][j] * weights_IH[j][i];
        }
        hidden_node_values[i] = tanh(hidden_node_values[i]);
    }

    //calculate the output of the network
    // the output neuron is linear
    obtained_output = 0.0;
    for (i = 0; i < num_hidden_nodes; i++) {
        obtained_output += hidden_node_values[i] * weights_HO[i];
    }

    //calculate the error
    desviation_error = obtained_output - train_output[num_train_inputs];

}


//************************************
//adjust the weights hidden - output
void backpropagate_error_HO(void)
{
    for (int k = 0; k < num_hidden_nodes; k++) {
        double update = learning_factor_HO * desviation_error * hidden_node_values[k];
        weights_HO[k] -= update;

        //regularisation on the output weights
        if (weights_HO[k] < -5) {
            weights_HO[k] = -5;
        } else if (weights_HO[k] > 5) {
            weights_HO[k] = 5;
        }
    }
}


//************************************
//adjust the weights input - hidden
void backpropagate_error_IH(void)
{
    int i;
    for (i = 0; i < num_hidden_nodes; i++) {
        for (int k = 0; k < num_input_nodes; k++) {
            double update = 1 - (hidden_node_values[i] * hidden_node_values[i]);
            update *= weights_HO[i] * desviation_error * learning_factor_IH;
            update *= train_inputs[num_train_inputs][k];
            weights_IH[k][i] = weights_IH[k][i] - update;
        }
    }
}


//************************************
//generates a random number
double drand(void)
{
    return ((double)rand()) / (double)RAND_MAX;
}



//************************************
//set weights to random numbers
void randomize_weights(void)
{
    int i;
    int j;
    for (j = 0; j < num_hidden_nodes; j++) {
        weights_HO[j] = (drand() - 0.5) / 2;
        printf("H -> O random weight = %f\n", weights_HO[j]);
        for (i = 0; i < num_input_nodes; i++) {
            weights_IH[i][j] = (drand() - 0.5) / 5;
            printf("I -> H random weight = %f\n", weights_IH[i][j]);
        }
    }

}


//************************************
//read in the data
void init_data(void)
{
    printf("initialising data\n");

    // the data here is the XOR data
    // it has been rescaled to the range
    // [-1][1]
    // an extra input valued 1 is also added
    // to act as the bias
    // the output must lie in the range -1 to 1

    train_inputs[0][0] = 1;
    train_inputs[0][1] = -1;
    train_inputs[0][2] = 1;
    //bias
    train_output[0] = 1;

    train_inputs[1][0] = -1;
    train_inputs[1][1] = 1;
    train_inputs[1][2] = 1;
    //bias
    train_output[1] = 1;

    train_inputs[2][0] = 1;
    train_inputs[2][1] = 1;
    train_inputs[2][2] = 1;
    //bias
    train_output[2] = -1;

    train_inputs[3][0] = -1;
    train_inputs[3][1] = -1;
    train_inputs[3][2] = 1;
    //bias
    train_output[3] = -1;

}


//************************************
//display results
void displayResults(void)
{
    int i;
    for (i = 0; i < num_ouput_nodes; i++) {
        num_train_inputs = i;
        calculate_error();
        printf("%d: real_output = %d obtained_output = %f\n",
            num_train_inputs + 1,
            train_output[num_train_inputs],
            obtained_output);
    }
}


//************************************
//calculate the overall error
void calculate_rms_error(void)
{
    int i;
    root_mean_square_error = 0.0;
    for (i = 0; i < num_ouput_nodes; i++) {
        num_train_inputs = i;
        calculate_error();
        root_mean_square_error += desviation_error * desviation_error;
    }
    root_mean_square_error /= num_ouput_nodes;
    root_mean_square_error = sqrt(root_mean_square_error);
}



int main(void)
{
    int i;
    int j;

    //seed random number function
    srand(time(NULL));

    //initiate the weights
    randomize_weights();

    //load in the data
    init_data();

    //train the network
    for (j = 0; j <= num_epoch; j++) {
        for (i = 0; i < num_ouput_nodes; i++) {
            //select a pattern at random
            num_train_inputs = rand() % num_ouput_nodes;

            //calculate the current network output
            // and error for this pattern
            calculate_error();

            //change network weights
            backpropagate_error_HO();
            backpropagate_error_IH();
        }

        //display the overall network error
        // after each epoch
        calculate_rms_error();

        printf("epoch = %d root_mean_square_error = %f\n", j, root_mean_square_error);
    }

    //training has finished
    // display the results
    displayResults();
    return 0;
}
