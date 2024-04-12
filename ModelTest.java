import java.util.Arrays;

public class ModelTest {
    public static void main(String[] args) {
        // Create a model
        Model model = new Model();

        double[][][] input = new double[64][64][3];

        // Forward pass through the model
        double[] output = model.forward(input);

        // Print the output
        System.out.println("Output: " + Arrays.toString(output));
    }
}