package layers;

import java.util.Random;

public class FullyConnectedLayer {
    private int inputSize;
    private int outputSize;
    private double[][] weights;
    private double[] biases;
    private Random random;

    public FullyConnectedLayer(int inputSize, int outputSize) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.random = new Random();

        // Initialize weights and biases with random values
        initializeWeights();
        initializeBiases();
    }

    private void initializeWeights() {
        weights = new double[inputSize][outputSize];
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                // Initialize weights with random values between -1 and 1
                weights[i][j] = random.nextDouble() * 2 - 1;
            }
        }
    }

    private void initializeBiases() {
        biases = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            // Initialize biases with random values between -1 and 1
            biases[i] = random.nextDouble() * 2 - 1;
        }
    }

    public double[] forward(double[] input) {
        double[] output = new double[outputSize];
        for (int j = 0; j < outputSize; j++) {
            double sum = 0;
            for (int i = 0; i < inputSize; i++) {
                sum += input[i] * weights[i][j];
            }
            output[j] = sum + biases[j];
        }
        return output;
    }
}
