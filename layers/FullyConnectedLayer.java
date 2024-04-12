package layers;

public class FullyConnectedLayer {
    private double[] weights;
    private double[] biases;

    public FullyConnectedLayer(int inputSize, int outputSize) {
        // Initialize weights and biases
        weights = new double[inputSize * outputSize];
        biases = new double[outputSize];
        initializeWeights(weights);
        initializeBiases(biases);
    }

    private void initializeWeights(double[] weights) {
        // Initialize weights with random values
        for (int i = 0; i < weights.length; i++) {
            weights[i] = Math.random() * 2 - 1;
        }
    }

    private void initializeBiases(double[] biases) {
        // Initialize biases with zeros
        for (int i = 0; i < biases.length; i++) {
            biases[i] = 0;
        }
    }

    public double[] forward(double[] input) {
        // Perform dot product and add biases
        double[] output = new double[biases.length];
        for (int i = 0; i < biases.length; i++) {
            double sum = 0;
            for (int j = 0; j < input.length; j++) {
                sum += weights[i * input.length + j] * input[j];
            }
            output[i] = sum + biases[i];
        }
        return output;
    }
}
