package layers;
import java.util.Arrays;

public class BatchNormalization {
    private int channels;
    private double[] mean;
    private double[] variance;
    private double[] gamma;
    private double[] beta;
    private double epsilon = 1e-5; // Small constant to prevent division by zero

    public BatchNormalization(int channels) {
        this.channels = channels;
        this.mean = new double[channels];
        this.variance = new double[channels];
        this.gamma = new double[channels];
        this.beta = new double[channels];

        // Initialize gamma and beta to 1 and 0 respectively
        Arrays.fill(gamma, 1.0);
        Arrays.fill(beta, 0.0);
    }

    public double[][][] forward(double[][][] input) {
        int height = input.length;
        int width = input[0].length;

        double[][][] output = new double[height][width][channels];

        // Compute mean and variance
        computeMean(input);
        computeVariance(input);

        // Normalize input
        for (int c = 0; c < channels; c++) {
            double std = Math.sqrt(variance[c] + epsilon);
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    output[i][j][c] = gamma[c] * (input[i][j][c] - mean[c]) / std + beta[c];
                }
            }
        }

        return output;
    }

    private void computeMean(double[][][] input) {
        int height = input.length;
        int width = input[0].length;

        for (int c = 0; c < channels; c++) {
            double sum = 0.0;
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    sum += input[i][j][c];
                }
            }
            mean[c] = sum / (height * width);
        }
    }

    private void computeVariance(double[][][] input) {
        int height = input.length;
        int width = input[0].length;

        for (int c = 0; c < channels; c++) {
            double sum = 0.0;
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    sum += Math.pow(input[i][j][c] - mean[c], 2);
                }
            }
            variance[c] = sum / (height * width);
        }
    }
}
