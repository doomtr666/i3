#include "fx_processor.h"

#include <filesystem>
#include <fstream>
#include <numeric>

std::string fx_processor::normalize_path(const std::string& path)
{
    return std::filesystem::weakly_canonical(std::filesystem::path(path).make_preferred()).string();
}

bool fx_processor::resolve(const std::string& base_path, const std::string& path, std::string& resolved_path)
{
    // default is local path
    resolved_path = normalize_path(std::filesystem::path(base_path).replace_filename(path).string());
    if (!std::filesystem::exists(resolved_path))
        return false;
    return true;
}

std::unique_ptr<char[]> fx_processor::read_file(const std::string& path)
{
    std::ifstream file(path);
    if (!file.is_open())
    {
        std::cerr << "failed to open file: " << path << std::endl;
        return nullptr;
    }

    file.seekg(0, std::ios::end);
    auto size = (uint32_t)file.tellg();
    file.seekg(0, std::ios::beg);

    auto buffer = std::make_unique<char[]>(size + 1);
    file.read(buffer.get(), size);
    buffer[size] = 0;

    return buffer;
}

bool fx_processor::parse_file(const std::string& path, std::vector<std::string>& imports)
{
    if (parsed_files_.find(path) != parsed_files_.end())
        return true;

    auto src = read_file(path);
    if (src == nullptr)
        return false;

    auto ast = parser_.parse(src.get(), path.c_str());

    if (ast == nullptr)
        return false;

    // process imports
    std::vector<std::string> new_imports;
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
            new_imports.push_back(resolved_path);
        }
    }

    // add file to the map
    parsed_files_[path] = files_.size();
    files_.push_back({path, std::move(src), ast, new_imports});

    // add new imports
    imports.insert(imports.end(), new_imports.begin(), new_imports.end());

    return true;
}

bool fx_processor::is_cyclic(int u,
                             std::vector<std::vector<int>>& adj,
                             std::vector<bool>& visited,
                             std::vector<bool>& recStack)
{
    // If the node is already in the recursion stack, a cycle is detected
    if (recStack[u])
    {
        std::cerr << "circular dependency detected involving:" << std::endl;
        for (int i = 0; i < recStack.size(); i++)
        {
            if (recStack[i])
                std::cerr << " " << files_[i].path << std::endl;
        }

        return true;
    }

    // If the node is already visited and not in recursion stack, no need to check again
    if (visited[u])
        return false;

    // Mark the current node as visited and add it to the recursion stack
    visited[u] = true;
    recStack[u] = true;

    // Recur for all neighbors
    for (int x : adj[u])
    {
        if (is_cyclic(x, adj, visited, recStack))
            return true;
    }

    // Remove the node from the recursion stack
    recStack[u] = false;
    return false;
}

bool fx_processor::order_files()
{
    const size_t num_files = files_.size();

    // Build adjacency list for dependencies
    std::vector<std::vector<int>> adj(num_files);
    for (size_t i = 0; i < num_files; ++i)
    {
        for (const auto& import : files_[i].imports)
        {
            auto it = parsed_files_.find(import);
            if (it == parsed_files_.end())
            {
                std::cerr << "internal error: imported file not found in parsed_files_ map." << std::endl;
                return false;
            }
            adj[i].push_back(it->second);
        }
    }

    // Detect cycles using DFS
    std::vector<bool> visited(num_files, false);
    std::vector<bool> recursion_stack(num_files, false);
    for (size_t i = 0; i < num_files; ++i)
    {
        if (!visited[i] && is_cyclic(static_cast<int>(i), adj, visited, recursion_stack))
        {
            return false;  // Cycle found
        }
    }

    // The current approach iteratively finds nodes with all dependencies met and adds them to ordered_
    std::vector<int> remaining_files(num_files);
    std::iota(remaining_files.begin(), remaining_files.end(), 0);  // Initialize with 0, 1, ..., num_files-1

    while (!remaining_files.empty())
    {
        bool found_next = false;
        for (auto it = remaining_files.begin(); it != remaining_files.end(); ++it)
        {
            int current_file_idx = *it;
            bool all_dependencies_met = true;
            for (int dependency_idx : adj[current_file_idx])
            {
                if (std::find(ordered_.begin(), ordered_.end(), dependency_idx) == ordered_.end())
                {
                    all_dependencies_met = false;
                    break;
                }
            }
            if (all_dependencies_met)
            {
                ordered_.push_back(current_file_idx);
                remaining_files.erase(it);
                found_next = true;
                break;
            }
        }
        if (!found_next && !remaining_files.empty())
        {
            // This should not happen if there are no cycles and the graph is valid
            std::cerr << "internal error: could not order remaining files, possibly due to an undetected cycle or "
                         "logic error."
                      << std::endl;
            return false;
        }
    }

    return true;
}

bool fx_processor::generate_parameters(fx_file& file)
{
    return true;
}

bool fx_processor::generate_shader(fx_file& file)
{
    for (auto& node : file.ast->nodes)
    {
        if (node->name == "slang")
        {
            // add line directive to have correct error report when compiling with slangc
            std::stringstream ss;
            ss << "#line " << node->line << " " << std::filesystem::path(file.path) << std::endl;

            // hack to capture the verbatim string, remove #slang  and #end tags
            auto src = node->token_to_string();
            src = src.substr(sizeof("#slang"), src.size() - sizeof("#slang") - sizeof("#end") - 1);
            ss << src << std::endl;

            shader_source_ += ss.str();
        }
    }

    return true;
}

bool fx_processor::generate_graphics_pipeline(std::shared_ptr<peg::Ast>& node)
{
    // name is first
    auto name = node->nodes[0]->token_to_string();

    // inherit
    std::vector<std::string> inherits;
    for (auto inherit : node->nodes[1]->nodes)
        inherits.push_back(inherit->token_to_string());

    for (uint32_t i = 2; i < node->nodes.size(); i++)
    {
        auto& stmt = node->nodes[i];

        if (stmt->name == "pipeline_var_stmt")
        {
        }
    }

    return true;
}

bool fx_processor::generate_compute_pipeline(std::shared_ptr<peg::Ast>& node)
{
    // name is first
    auto name = node->nodes[0]->token_to_string();

    return true;
}

bool fx_processor::generate_pipelines(fx_file& file)
{
    // extract pipeline

    for (auto& node : file.ast->nodes)
    {
        if (node->name == "graphics" || node->name == "compute")
        {
            if (node->name == "graphics")
                if (!generate_graphics_pipeline(node))
                    return false;
            if (node->name == "compute")
                if (!generate_compute_pipeline(node))
                    return false;
        }
    }

    return true;
}

bool fx_processor::generate()
{
    for (auto file_index : ordered_)
    {
        auto& file = files_[file_index];

        if (!generate_parameters(file))
            return false;

        if (!generate_shader(file))
            return false;

        if (!generate_pipelines(file))
            return false;
    }

    if (debug_)
    {
        std::cout << "shader source:" << std::endl;
        std::cout << "->" << shader_source_ << "<-" << std::endl;
    }

    return true;
}

bool fx_processor::process(const std::string& path)
{
    // parse all involved files, but only once
    std::vector<std::string> imports;
    imports.push_back(normalize_path(path));

    while (!imports.empty())
    {
        auto path = imports.back();
        imports.pop_back();

        if (!parse_file(path, imports))
            return false;
    }

    // order files
    if (!order_files())
        return false;

    if (debug_)
    {
        for (auto i : ordered_)
        {
            auto& file = files_[i];

            std::cout << "file: " << file.path << std::endl;
            std::cout << peg::ast_to_s(file.ast) << std::endl;
        }
    }

    // generate flat buffer
    if (!generate())
        return false;

    return true;
}