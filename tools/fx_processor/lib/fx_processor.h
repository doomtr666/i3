#pragma once

#include "parser.h"

#include <filesystem>

class fx_processor
{
    parser parser_;

    struct fx_file
    {
        std::string path;
        std::unique_ptr<char[]> content;
        std::shared_ptr<peg::Ast> ast;
        std::vector<std::string> imports;
    };

    std::map<std::string, int> parsed_files_;
    std::vector<fx_file> files_;
    std::vector<int> ordered_;

    std::string shader_source_;

    bool debug_ = false;

    std::string normalize_path(const std::string& path);
    bool resolve(const std::string& base_path, const std::string& path, std::string& resolved_path);
    std::unique_ptr<char[]> read_file(const std::string& path);
    bool parse_file(const std::string& path, std::vector<std::string>& imports);
    bool is_cyclic(int u, std::vector<std::vector<int>>& adj, std::vector<bool>& visited, std::vector<bool>& recStack);
    bool order_files();

    bool generate_parameters(fx_file& file);
    bool generate_shader(fx_file& file);
    bool generate_graphics_pipeline(std::shared_ptr<peg::Ast>& node);
    bool generate_compute_pipeline(std::shared_ptr<peg::Ast>& node);

    bool generate_pipelines(fx_file& file);
    bool generate();

  public:
    void enable_debug() { debug_ = true; }

    bool process(const std::string& path);
};