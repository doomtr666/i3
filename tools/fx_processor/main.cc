#include "parser.h"

#include <deque>
#include <filesystem>
#include <fstream>

class fx_processor
{
    struct fx_file
    {
        std::string path;
        std::shared_ptr<peg::Ast> ast;
    };

    std::deque<fx_file> files_;
    bool debug_ = false;
    int import_counter_ = 0;

    bool read_file(const std::string& path, std::string& content)
    {
        std::ifstream file(path);
        if (!file.is_open())
        {
            std::cerr << "failed to open file: " << path << std::endl;
            return "";
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        content = buffer.str();
    }

    bool resolve(const std::string& base_path, const std::string& path, std::string& resolved_path)
    {
        // default is local path
        resolved_path = std::filesystem::path(base_path).replace_filename(path).generic_string();
        if (!std::filesystem::exists(resolved_path))
            return false;
        return true;
    }

  public:
    void enable_debug() { debug_ = true; }

    bool process_file(const std::string& path)
    {
        parser parser_;

        std::string src;
        if (!read_file(path, src))
            return false;

        auto ast = parser_.parse(src.c_str(), path.c_str());

        if (ast == nullptr)
            return false;

        if (debug_)
        {
            std::cout << "file: " << path << std::endl;
            std::cout << peg::ast_to_s(ast) << std::endl;
        }

        files_.push_front({path, ast});

        // process imports
        std::deque<std::string> imports;
        for (auto& node : ast->nodes)
        {
            if (node->name == "import")
            {
                auto import_file = node->nodes[0];
                auto import_file_str = import_file->token_to_string();

                std::string resolved_path;
                if (!resolve(path, import_file_str, resolved_path))
                {
                    parser_.error(node->nodes[0], "failed to open import file: " + import_file_str);
                    return false;
                }

                imports.push_front(resolved_path);
            }
        }

        for (auto& import : imports)
        {
            if (!process_file(import))
                return false;
        }

        return true;
    }
};

int main(int argc, char** argv)
{
    const char* path = argv[1];

    fx_processor processor;
    processor.enable_debug();

    if (!processor.process_file(path))
    {
        std::cerr << "failed to process file: " << path << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "compilation succeeded" << std::endl;
    return EXIT_SUCCESS;
}