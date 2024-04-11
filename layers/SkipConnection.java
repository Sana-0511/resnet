package layers;

public class SkipConnection {
    public double[][][] forward(double[][][] input, double[][][] residual) {
        // Perform element-wise addition of input and residual
        int depth = input.length;
        int height = input[0].length;
        int width = input[0][0].length;

        double[][][] output = new double[depth][height][width];

        for (int d = 0; d < depth; d++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    output[d][h][w] = input[d][h][w] + residual[d][h][w];
                }
            }
        }

        return output;
    }
}

