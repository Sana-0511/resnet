package layers;

import java.util.Random;

public class FullyConnectedLayer {
    private int inputSize;
    private int outputSize;
    private double[][] weights;
    private double[] biases;

    public FullyConnectedLayer(int inputSize, int outputSize) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.weights = new double[outputSize][inputSize];
        this.biases = new double[outputSize];

        // Initialize weights and biases randomly
        initializeWeights();
        initializeBiases();
    }

    private void initializeWeights() {
        Random random = new Random();
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                weights[i][j] = random.nextDouble() * 0.01; // Initialize weights randomly
            }
        }
    }

    private void initializeBiases() {
        Random random = new Random();
        for (int i = 0; i < outputSize; i++) {
            biases[i] = random.nextDouble() * 0.01; // Initialize biases randomly
        }
    }

    public double[] forward(double[] input) {
        double[] output = new double[outputSize];

        // Perform matrix multiplication: output = weights * input + biases
        for (int i = 0; i < outputSize; i++) {
            double sum = 0.0;
            for (int j = 0; j < inputSize; j++) {
                sum += weights[i][j] * input[j];
            }
            output[i] = sum + biases[i];
        }

        return output;
    }
}

