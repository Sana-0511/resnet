package layers;

import java.util.Random;

public class ConvolutionLayer {
    private int inputChannels;
    private int outputChannels;
    private int kernelSize;
    private double[][][][] weights;
    private double[] biases;

    public ConvolutionLayer(int inputChannels, int outputChannels, int kernelSize) {
        this.inputChannels = inputChannels;
        this.outputChannels = outputChannels;
        this.kernelSize = kernelSize;
        this.weights = new double[outputChannels][inputChannels][kernelSize][kernelSize];
        this.biases = new double[outputChannels];

        // Initialize weights and biases randomly
        initializeWeights();
        initializeBiases();
    }

    private void initializeWeights() {
        Random random = new Random();
        for (int i = 0; i < outputChannels; i++) {
            for (int j = 0; j < inputChannels; j++) {
                for (int k = 0; k < kernelSize; k++) {
                    for (int l = 0; l < kernelSize; l++) {
                        weights[i][j][k][l] = random.nextDouble() * 0.01; // Initialize weights randomly
                    }
                }
            }
        }
    }

    private void initializeBiases() {
        Random random = new Random();
        for (int i = 0; i < outputChannels; i++) {
            biases[i] = random.nextDouble() * 0.01; // Initialize biases randomly
        }
    }

    public double[][][] forward(double[][][] input) {
        int inputSize = input.length;
        int outputSize = inputSize - kernelSize + 1;
        double[][][] output = new double[outputSize][outputSize][outputChannels];

        for (int i = 0; i < outputChannels; i++) {
            for (int x = 0; x < outputSize; x++) {
                for (int y = 0; y < outputSize; y++) {
                    double sum = biases[i];
                    for (int j = 0; j < inputChannels; j++) {
                        for (int k = 0; k < kernelSize; k++) {
                            for (int l = 0; l < kernelSize; l++) {
                                // Check bounds to prevent ArrayIndexOutOfBoundsException
                                if (x + k < inputSize && y + l < inputSize) {
                                    sum += input[x + k][y + l][j] * weights[i][j][k][l];
                                }
                            }
                        }
                    }
                    output[x][y][i] = sum;
                }
            }
        }
        return output;
    }
}
