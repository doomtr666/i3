#include "fx_processor.h"

#include <filesystem>
#include <fstream>

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
    // get edges
    std::vector<std::pair<int, int>> edges;
    for (uint32_t i = 0; i < files_.size(); i++)
    {
        for (auto& import : files_[i].imports)
        {
            auto it = parsed_files_.find(import);
            if (it == parsed_files_.end())
            {
                // should not happen
                std::cerr << "internal error" << std::endl;
                return false;
            }
            edges.push_back({i, it->second});
        }
    }

    // adjencency
    std::vector<std::vector<int>> adj(files_.size());
    for (auto& edge : edges)
        adj[edge.first].push_back(edge.second);

    std::vector<bool> visited(files_.size(), false);
    std::vector<bool> recStack(files_.size(), false);

    // Check for cycles starting from every unvisited node
    for (int i = 0; i < files_.size(); i++)
    {
        if (!visited[i] && is_cyclic(i, adj, visited, recStack))
            return false;  // Cycle found
    }

    // no cycles, get a correct order
    std::vector<int> unordered;

    for (uint32_t i = 0; i < files_.size(); i++)
        unordered.push_back(i);

    while (!unordered.empty())
    {
        for (uint32_t i = 0; i < unordered.size(); i++)
        {
            bool all_req = true;

            for (uint32_t j = 0; j < adj[unordered[i]].size(); j++)
            {
                auto it = std::find(ordered_.begin(), ordered_.end(), adj[unordered[i]][j]);
                if (it == ordered_.end())
                {
                    all_req = false;
                    break;
                }
            }

            if (all_req)
            {
                ordered_.push_back(unordered[i]);
                unordered.erase(unordered.begin() + i);
                break;
            }
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

bool fx_processor::generate_pipelines(fx_file& file)
{
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