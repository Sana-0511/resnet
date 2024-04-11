package layers;

public class ReLU {
    public double[][][] forward(double[][][] input) {
        int depth = input.length;
        int height = input[0].length;
        int width = input[0][0].length;

        double[][][] output = new double[depth][height][width];

        for (int d = 0; d < depth; d++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    output[d][h][w] = Math.max(0, input[d][h][w]); // ReLU function: max(0, x)
                }
            }
        }

        return output;
    }
}

