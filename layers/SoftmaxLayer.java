package layers;

public class SoftmaxLayer {
    public double[] forward(double[] input) {
        double[] output = new double[input.length];
        double sum = 0.0;

        // Compute exponentials and sum
        for (int i = 0; i < input.length; i++) {
            output[i] = Math.exp(input[i]);
            sum += output[i];
        }

        // Normalize using the sum
        for (int i = 0; i < input.length; i++) {
            output[i] /= sum;
        }

        return output;
    }
}
