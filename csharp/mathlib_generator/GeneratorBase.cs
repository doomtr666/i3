using System.IO;

abstract class GeneratorBase
{
    StreamWriter writer_;

    public void WriteIdent(int ident)
    {
        for (int i = 0; i < ident; ++i)
            writer_.Write("    ");
    }

    public void WriteLine(string line="", int ident = 0)
    {
        WriteIdent(ident);
        writer_.WriteLine(line);
    }

    public void Write(string line, int ident = 0)
    {
        WriteIdent(ident);
        writer_.Write(line);
    }

    public void Run(string filename)
    {
        using (writer_ = new StreamWriter(filename))
        {
            Generate();
        }
    }

    public abstract void Generate();
}