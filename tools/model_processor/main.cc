#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/Importer.hpp>

#include <iostream>

class options
{
    std::string input_file_;
    std::string output_file_;

    char** get_cmd_option(char** begin, char** end, const std::string& option)
    {
        char** itr = std::find(begin, end, option);
        if (itr != end && ++itr != end)
        {
            return itr;
        }
        return 0;
    }

  public:
    bool parse(int argc, char** argv)
    {
        auto begin = argv + 1;
        auto end = argv + argc;

        auto iter = get_cmd_option(begin, end, "-i");
        if (!iter)
        {
            std::cerr << "Input file not specified. Use -i <model_file>\n";
            return false;
        }
        if (iter == end)
        {
            std::cerr << "Input file not specified. Use -i <model_file>\n";
            return false;
        }
        input_file_ = *iter;

        iter = get_cmd_option(argv, argv + argc, "-o");
        if (!iter)
        {
            std::cerr << "Output file not specified. Use -o <output_file>\n";
            return false;
        }
        if (iter == end)
        {
            std::cerr << "Output file not specified. Use -o <output_file>\n";
            return false;
        }

        output_file_ = *iter;

        return true;
    }

    const std::string& input_file() const { return input_file_; }
    const std::string& output_file() const { return output_file_; }
};

int main(int argc, char** argv)
{
    // get arguments
    options opts;
    if (!opts.parse(argc, argv))
    {
        std::cerr << "Usage: model_processor -i <input_file> -o <output_file>\n";
        return EXIT_FAILURE;
    }

    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(
        opts.input_file().c_str(), aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_JoinIdenticalVertices);

    if (!scene)
    {
        std::cerr << "Error loading model: " << importer.GetErrorString() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}