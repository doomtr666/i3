#include "parser.h"

#include <deque>
#include <filesystem>
#include <fstream>

class fx_processor
{
    parser parser_;

    struct fx_file
    {
        std::string path;
        std::string content;
        std::shared_ptr<peg::Ast> ast;
    };

    std::map<std::string, fx_file> files_;

    bool debug_ = false;

    bool read_file(const std::string& path, std::string& content)
    {
        std::ifstream file(path);
        if (!file.is_open())
        {
            std::cerr << "failed to open file: " << path << std::endl;
            return false;
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        content = buffer.str();

        return true;
    }

    bool resolve(const std::string& base_path, const std::string& path, std::string& resolved_path)
    {
        // default is local path
        auto target =
            std::filesystem::weakly_canonical(std::filesystem::path(base_path).replace_filename(path).make_preferred());

        resolved_path = target.string();

        if (!std::filesystem::exists(resolved_path))
            return false;
        return true;
    }

    bool parse_file(const std::string& path, std::vector<std::string>& imports)
    {
        if (files_.find(path) != files_.end())
            return true;

        std::string src;
        if (!read_file(path, src))
            return false;

        auto ast = parser_.parse(src.c_str(), path.c_str());

        if (ast == nullptr)
            return false;

        // process imports
        for (auto& node : ast->nodes)
        {
            if (node->name == "import")
            {
                auto import_file = node->nodes[0];
                auto import_file_str = import_file->token_to_string();

                std::string resolved_path;
                if (!resolve(path, import_file_str, resolved_path))
                {
                    parser_.error(import_file, "failed to open import file: " + import_file_str);
                    return false;
                }
                imports.push_back(resolved_path);
            }
        }

        // add file to the map
        files_[path] = {path, src, ast};
    }

  public:
    void enable_debug() { debug_ = true; }

    bool process(const std::string& path)
    {
        std::vector<std::string> imports;
        imports.push_back(path);

        while (!imports.empty())
        {
            auto path = imports.back();
            imports.pop_back();

            if (!parse_file(path, imports))
                return false;
        }

        if (debug_)
        {
            for (auto& [path, file] : files_)
            {
                std::cout << "file: " << path << std::endl;
                std::cout << peg::ast_to_s(file.ast) << std::endl;
            }
        }

        return true;
    }
};

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