package layers;

public class GlobalAveragePooling {
    public double[] forward(double[][][] input) {
        int depth = input.length;
        int height = input[0].length;
        int width = input[0][0].length;

        double[] output = new double[depth];

        // Compute average pooling across height and width dimensions
        for (int d = 0; d < depth; d++) {
            double sum = 0.0;
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    sum += input[d][h][w];
                }
            }
            output[d] = sum / (height * width);
        }

        return output;
    }
}

