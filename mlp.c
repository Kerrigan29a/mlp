/*
 * Copyright (c) 2004  John Bullinaria
 * Copyright (c) 2014  Javier Escalada Gómez (kerrigan29a@gmail.com)
 */

/*
 * For explanations see:  http://www.cs.bham.ac.uk/~jxb/NN/nn.html
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define NUM_TRAIN_INPUTS    4

#define NUM_INPUT_UNITS     2
#define NUM_HIDDEN_UNITS    4
#define NUM_OUTPUT_UNITS    1

#define MAX_EPOCH           1000000

/*
 * This is a good approximation to a uniform random distribution in the
 * interval [0, 1):
 * http://www.thinkage.ca/english/gcos/expl/c/lib/rand.html
 */
#define normalized_rand() ((double) rand() / (RAND_MAX + 1))



/*
 * This function generates values with the interval [-limit, limit)
 */
#define ranged_rand(limit) (2.0 * ( normalized_rand() - 0.5 ) * (limit))



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
    double eta = 0.5;
    double alpha = 0.9;
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

    /* Train */
    for( epoch = 0; epoch < MAX_EPOCH; epoch++ ) {

        /* Randomize order of individuals */
        for( p = 1; p <= NumPattern; p++ ) {
            ranpat[p] = p;
        }
        for( p = 1; p <= NumPattern; p++ ) {
            np = p + normalized_rand() * ( NumPattern + 1 - p );
            op = ranpat[p];
            ranpat[p] = ranpat[np];
            ranpat[np] = op;
        }

        Error = 0.0;
        /* Repeat for all the training patterns */
        for( np = 1; np <= NumPattern; np++ ) {
            p = ranpat[np];

            /* Compute hidden unit activations */
            for( j = 1; j <= NumHidden; j++ ) {
                SumH[p][j] = WeightIH[0][j];
                for( i = 1; i <= NumInput; i++ ) {
                    SumH[p][j] += Input[p][i] * WeightIH[i][j];
                }
                Hidden[p][j] = 1.0/(1.0 + exp(-SumH[p][j]));
            }

            /* Compute output unit activations and errors */
            for( k = 1; k <= NumOutput; k++ ) {
                SumO[p][k] = WeightHO[0][k];
                for( j = 1; j <= NumHidden; j++ ) {
                    SumO[p][k] += Hidden[p][j] * WeightHO[j][k];
                }
                /* Sigmoidal Outputs */
                Output[p][k] = 1.0/(1.0 + exp(-SumO[p][k]));
                /* Linear Outputs */
                // Output[p][k] = SumO[p][k];

                Error += 0.5 * (Target[p][k] - Output[p][k]) * (Target[p][k] - Output[p][k]);   /* SSE */
                /* Cross-Entropy Error */
                // Error -= ( Target[p][k] * log( Output[p][k] ) + ( 1.0 - Target[p][k] ) * log( 1.0 - Output[p][k] ) );

                /* Sigmoidal Outputs */
                DeltaO[k] = (Target[p][k] - Output[p][k]) * Output[p][k] * (1.0 - Output[p][k]);   /* SSE */
                /* Sigmoidal Outputs, Cross-Entropy Error */
                // DeltaO[k] = Target[p][k] - Output[p][k];
                /* Linear Outputs */
                // DeltaO[k] = Target[p][k] - Output[p][k]; /* SSE */
            }

            /* Back-propagate errors to hidden layer */
            for( j = 1; j <= NumHidden; j++ ) {
                SumDOW[j] = 0.0;
                for( k = 1; k <= NumOutput; k++ ) {
                    SumDOW[j] += WeightHO[j][k] * DeltaO[k];
                }
                DeltaH[j] = SumDOW[j] * Hidden[p][j] * (1.0 - Hidden[p][j]);
            }

            /* Update weights WeightIH */
            for( j = 1; j <= NumHidden; j++ ) {
                DeltaWeightIH[0][j] = eta * DeltaH[j] + alpha * DeltaWeightIH[0][j];
                WeightIH[0][j] += DeltaWeightIH[0][j];
                for( i = 1; i <= NumInput; i++ ) {
                    DeltaWeightIH[i][j] = eta * Input[p][i] * DeltaH[j] + alpha * DeltaWeightIH[i][j];
                    WeightIH[i][j] += DeltaWeightIH[i][j];
                }
            }

            /* Update weights WeightHO */
            for( k = 1; k <= NumOutput; k ++ ) {
                DeltaWeightHO[0][k] = eta * DeltaO[k] + alpha * DeltaWeightHO[0][k];
                WeightHO[0][k] += DeltaWeightHO[0][k];
                for( j = 1; j <= NumHidden; j++ ) {
                    DeltaWeightHO[j][k] = eta * Hidden[p][j] * DeltaO[k] + alpha * DeltaWeightHO[j][k];
                    WeightHO[j][k] += DeltaWeightHO[j][k];
                }
            }
        }

        if( epoch % 1000 == 0 ) {
            fprintf(stdout, "\rEpoch %-6d :   Error = %f", epoch, Error);
        }

        /* Stop learning when 'near enough' */
        if( Error < 0.000001 ) {
            fprintf(stdout, "\rEpoch %-6d :   Error = %f - DONE!\n", epoch, Error);
            goto print_result;
        }
    }
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

/*******************************************************************************/
