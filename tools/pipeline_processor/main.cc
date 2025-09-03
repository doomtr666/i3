#include "parser.h"

int main(int argc, char** argv)
{
    parser p;

    const char* src = R"(

    pipeline test_pipeline {
        compile(vs, vertexMain);
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