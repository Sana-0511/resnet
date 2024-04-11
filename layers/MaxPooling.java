package layers;

public class MaxPooling {
    private int poolSize;
    private int stride;

    public MaxPooling(int poolSize, int stride) {
        this.poolSize = poolSize;
        this.stride = stride;
    }

    public double[][][] forward(double[][][] input) {
        int inputDepth = input.length;
        int inputHeight = input[0].length;
        int inputWidth = input[0][0].length;

        int outputDepth = inputDepth;
        int outputHeight = (inputHeight - poolSize) / stride + 1;
        int outputWidth = (inputWidth - poolSize) / stride + 1;

        double[][][] output = new double[outputDepth][outputHeight][outputWidth];

        for (int d = 0; d < outputDepth; d++) {
            for (int h = 0; h < outputHeight; h++) {
                for (int w = 0; w < outputWidth; w++) {
                    // Find max value in the pool
                    double maxVal = Double.NEGATIVE_INFINITY;
                    for (int i = 0; i < poolSize; i++) {
                        for (int j = 0; j < poolSize; j++) {
                            double val = input[d][h * stride + i][w * stride + j];
                            if (val > maxVal) {
                                maxVal = val;
                            }
                        }
                    }
                    output[d][h][w] = maxVal;
                }
            }
        }

        return output;
    }
}

