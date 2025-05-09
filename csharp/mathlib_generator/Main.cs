using System;
using System.IO;

class Program
{
    public static void Main(params string[] args)
    {
        if (args.Length != 1)
        {
            Console.WriteLine("Usage: mathlib_generator <output_directory>");
            return;
        }

        string outputDirectory = args[0];

        // Ensure the output directory exists
        if (!Directory.Exists(outputDirectory))
            Directory.CreateDirectory(outputDirectory);

        // Generate the math library files
        Console.WriteLine("Generating mathlib...");

        var generator = new VectorGenerator();
        generator.Run(Path.Join(outputDirectory, "vec.h"));

        var matrixGenerator = new MatrixGenerator();
        matrixGenerator.Run(Path.Join(outputDirectory, "mat.h"));

        Console.WriteLine("mathlib generated successfully.");
    }
}
