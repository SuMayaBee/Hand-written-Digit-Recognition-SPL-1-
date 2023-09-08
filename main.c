#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#include "include/data_file.h"
#include "include/neural_network.h"

// Convert a pixel value from 0-255 to one from 0 to 1
#define PIXEL_SCALE(x) (((float)(x)) / 255.0f)

#define STEPS 10
#define BATCH_SIZE 100

const char *train_images_file = "data-set/train-images-idx3-ubyte";
const char *train_labels_file = "data-set/train-labels-idx1-ubyte";
const char *test_images_file = "data-set/t10k-images-idx3-ubyte";
const char *test_labels_file = "data-set/t10k-labels-idx1-ubyte";

data_dataset_t *train_dataset, *test_dataset;

float calculate_accuracy(data_dataset_t *dataset, neural_network_t *network)
{
    float activations[DATA_LABELS], max_activation;
    int i, j, correct, predict;

    // Loop through the dataset
    for (i = 0, correct = 0; i < dataset->size; i++)
    {
        // Calculate the activations for each image using the neural network
        neural_network_hypothesis(&dataset->images[i], network, activations);

        // Set predict to the index of the greatest activation
        for (j = 0, predict = 0, max_activation = activations[0]; j < DATA_LABELS; j++)
        {
            if (max_activation < activations[j])
            {
                max_activation = activations[j];
                predict = j;
            }
        }

        // Increment the correct count if we predicted the right label
        if (predict == dataset->labels[i])
        {
            correct++;
        }
    }

    // Return the percentage we predicted correctly as the accuracy
    return ((float)correct) / ((float)dataset->size);
}

void predict_single_image(data_image_t *image, int n, neural_network_t *network)
{
    float activations[DATA_LABELS], max_activation;
    int j, predict;

    // Calculate the activations for the image using the neural network
    neural_network_hypothesis(image, network, activations);

    // Set predict to the index of the greatest activation
    for (j = 0, predict = 0, max_activation = activations[0]; j < DATA_LABELS; j++)
    {
        if (max_activation < activations[j])
        {
            max_activation = activations[j];
            predict = j;
        }
    }

    printf("Actual Label: %d, Predicted Label: %d\n", n, predict);
    printf("\n");
    if (n == predict)
    {
        printf("Congratulation! Correct prediction.\n\n\n");
    }
    else
    {
        printf("Oops! You failed to predict. \n\n\n");
    }
}

int main(int argc, char **argv)
{

    data_dataset_t batch;
    neural_network_t network;
    float loss, accuracy;
    int batches, steps;
    uint32_t i, j, k;

    train_dataset = data_get_dataset(train_images_file, train_labels_file);
    test_dataset = data_get_dataset(test_images_file, test_labels_file);

    if (NULL == train_dataset && NULL == test_dataset)
    {
        fprintf(stderr, "Could not load MNIST dataset\n");
        return 1;
    }

    neural_network_random_weights(&network);

    // Calculate how many batches (so we know when to wrap around)
    batches = train_dataset->size / BATCH_SIZE;
    printf("\n");

    printf("How many steps do you wanna train the neural network?\n");
    scanf("%d", &steps);
    printf("\n");

    for (int i = 0; i < steps; i++)
    {
        // Initialise a new batch
        data_batch(train_dataset, &batch, 100, i % batches);

        // Run one step of gradient descent and calculate the loss
        loss = neural_network_training_step(&batch, &network, 0.5);

        // Calculate the accuracy using the whole test dataset
        accuracy = calculate_accuracy(test_dataset, &network);

        printf("Step %04d\tAverage Loss: %.2f\tAccuracy: %.3f\n", i, loss / batch.size, accuracy);
    }

    printf("\n");
    int a;
    printf("\n");
    printf("Which serial of the data-set do you wanna test with?\n ");
    scanf("%d", &a);

    printf("\n The image is: \n\n");
    for (j = 0; j < DATA_IMAGE_HEIGHT; j++)
    {
        for (k = 0; k < DATA_IMAGE_WIDTH; k++)
        {
            printf("%1.1f ", PIXEL_SCALE(test_dataset->images[a].pixels[j * DATA_IMAGE_WIDTH + k]));
        }

        printf("\n");
    }

    printf("\n");
    predict_single_image(&test_dataset->images[a], test_dataset->labels[a], &network);

    data_free_dataset(train_dataset);
    data_free_dataset(test_dataset);

    return 0;
}
