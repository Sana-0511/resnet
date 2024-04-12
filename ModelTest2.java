import java.util.Arrays;

public class ModelTest2 {
    public static void main(String[] args) {
        // Create a model
        Model model = new Model();

        double[][][] input = new double[64][64][3];
        for (int i = 0; i < 64; i++) {
            for (int j = 0; j < 64; j++) {
                input[i][j][0] = Math.random();
                input[i][j][1] = Math.random();
                input[i][j][2] = Math.random();
            }
        }

        // Forward pass through the model
        double[] output = model.forward(input);

        // Print the output
        System.out.println("Output: " + Arrays.toString(output));
    }
}