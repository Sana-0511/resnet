package layers;

public class ResidualBlock {
    private ConvolutionLayer convLayer1;
    private ConvolutionLayer convLayer2;
    private BatchNormalization batchNorm1;
    private BatchNormalization batchNorm2;
    private ReLU relu;

    public ResidualBlock(int inputChannels, int outputChannels, int kernelSize) {
        // Define the layers within the residual block
        convLayer1 = new ConvolutionLayer(inputChannels, outputChannels, kernelSize);
        batchNorm1 = new BatchNormalization(outputChannels);
        relu = new ReLU();
        convLayer2 = new ConvolutionLayer(outputChannels, outputChannels, kernelSize);
        batchNorm2 = new BatchNormalization(outputChannels);
    }

    public double[][][] forward(double[][][] input) {
        // Perform forward pass through the residual block
        double[][][] output = convLayer1.forward(input);
        output = batchNorm1.forward(output);
        output = relu.forward(output);
        output = convLayer2.forward(output);
        output = batchNorm2.forward(output);

        // Add input to output (identity shortcut)
        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < input[0].length; j++) {
                for (int k = 0; k < input[0][0].length; k++) {
                    output[i][j][k] += input[i][j][k];
                }
            }
        }

        output = relu.forward(output); // Apply ReLU activation
        return output;
    }
}

