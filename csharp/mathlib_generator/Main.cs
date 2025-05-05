using System;

class Program
{
    public static void Main()
    {
        VectorGenerator generator = new VectorGenerator();
        generator.Run("vec.h");

        MatrixGenerator matrixGenerator = new MatrixGenerator();
        matrixGenerator.Run("mat.h");

        Console.WriteLine("mathlib generated successfully.");
    }
}