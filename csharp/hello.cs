using System;
using System.IO;
using System.Runtime.InteropServices;

class Dll
{
    // Import the C function from the shared library
    [DllImport("native/add_shared", EntryPoint="add", CallingConvention = System.Runtime.InteropServices.CallingConvention.Cdecl)]
    public static extern int Add(int a, int b);
}

class Program
{
    public static void Main()
    {
        Console.WriteLine("{0}", Directory.GetCurrentDirectory());

        Console.WriteLine("+++ Hello, World! +++");
        int r = Dll.Add(1, 2);
        Console.WriteLine($"1 + 2 = {r}");
        
    }
}
