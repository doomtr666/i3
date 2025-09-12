#include "lib/fx_processor.h"

int main(int argc, char** argv)
{
    const char* path = argv[1];

    fx_processor processor;
    processor.enable_debug();

    if (!processor.process(path))
    {
        std::cerr << "failed to process file: " << path << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "compilation succeeded" << std::endl;
    return EXIT_SUCCESS;
}