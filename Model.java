import layers.ConvolutionLayer;

import java.util.Arrays;

import layers.BatchNormalization;
import layers.ReLU;
import layers.ResidualBlock;
import layers.GlobalAveragePooling;
import layers.FullyConnectedLayer;

public class Model {
    private ConvolutionLayer conv1;
    private BatchNormalization batchNorm1;
    private ReLU relu;
    private ResidualBlock[] residualBlocks;
    private GlobalAveragePooling globalAvgPooling;
    private FullyConnectedLayer fullyConnectedLayer;

    // Define input parameters
    private int inputChannels = 3; // Adjust input channels accordingly
    private int kernelSize = 3; // Adjust kernel size accordingly
    private int numClasses = 10; // Adjust number of classes accordingly
    private int inputSize = 64; // Example input size

    public Model() {
        // Define the layers
        conv1 = new ConvolutionLayer(inputChannels, 64, kernelSize);
        batchNorm1 = new BatchNormalization(64);
        relu = new ReLU();

        // Create 4 residual blocks with 2 convolutional layers each
        residualBlocks = new ResidualBlock[4];
        for (int i = 0; i < 4; i++) {
            residualBlocks[i] = new ResidualBlock(64, 64, kernelSize);
        }

        globalAvgPooling = new GlobalAveragePooling();
        fullyConnectedLayer = new FullyConnectedLayer(64, numClasses); // Adjust input size and output size accordingly
    }

    public double[] forward(double[][][] input) {
        // Perform forward pass through the model
        double[][][] output = conv1.forward(input);
        output = batchNorm1.forward(output);
        output = relu.forward(output);

        // Pass through residual blocks
        for (int i = 0; i < 4; i++) {
            output = residualBlocks[i].forward(output);
        }

        // Global average pooling
        double[] pooledOutput = globalAvgPooling.forward(output);

        // Fully connected layer
        return fullyConnectedLayer.forward(pooledOutput);
    }

    public static void main(String[] args) {
    // Example usage
    Model model = new Model();
    double[][][] input = new double[model.inputSize][model.inputSize][model.inputChannels]; // Adjust input size and input channels accordingly
    double[] output = model.forward(input);
    System.out.println("Output: " + Arrays.toString(output));
    // Do something with the output...
}

}
