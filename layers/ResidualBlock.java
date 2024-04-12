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
        double[][][] residual = convLayer1.forward(input); // First convolutional layer
        residual = batchNorm1.forward(residual);
        residual = relu.forward(residual);

        residual = convLayer2.forward(residual); // Second convolutional layer
        residual = batchNorm2.forward(residual);

        // Add input to residual (identity shortcut)
        int depth = input.length;
        int height = input[0].length;
        int width = input[0][0].length;
        double[][][] shortcut = new double[depth][height][width];
        for (int i = 0; i < depth; i++) {
            for (int j = 0; j < height; j++) {
                for (int k = 0; k < width; k++) {
                    shortcut[i][j][k] = input[i][j][k];
                }
            }
        }
        residual = add(residual, shortcut);

        // Apply ReLU activation
        return relu.forward(residual);
    }

    private double[][][] add(double[][][] a, double[][][] b) {
        int depth = a.length;
        int height = a[0].length;
        int width = a[0][0].length;
        double[][][] result = new double[depth][height][width];
        for (int i = 0; i < depth; i++) {
            for (int j = 0; j < height; j++) {
                for (int k = 0; k < width; k++) {
                    result[i][j][k] = a[i][j][k] + b[i][j][k];
                }
            }
        }
        return result;
    }
}