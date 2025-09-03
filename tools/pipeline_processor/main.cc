#include "parser.h"

int main(int argc, char** argv)
{
    parser p;

    const char* src = R"(

    #slang
    #end

    // test pipeline
    pipeline  test_pipeline {  
        /* shaders */
        compile ( vs , vertexMain);   // vertex shader
        compile( ps , pixelMain);    // pixel shader
    }

    pipeline  test_pipeline2 {
    }

    )";

    auto ast = p.parse(src);

    if (ast == nullptr)
    {
        std::cerr << "compilation failed" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << peg::ast_to_s(ast) << std::endl;
    std::cout << "compilation succeeded" << std::endl;

    return EXIT_SUCCESS;
}