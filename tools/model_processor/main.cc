#include "fbs/model_generated.h"

#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/DefaultLogger.hpp>
#include <assimp/Importer.hpp>

#include <fstream>
#include <iostream>

class options
{
    std::string input_file_;
    std::string output_file_;
    bool verbose_ = false;

    char** get_cmd_option(char** begin, char** end, const std::string& option)
    {
        char** itr = std::find(begin, end, option);
        if (itr != end)
            return itr;
        return nullptr;
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
        if (++iter == end)
        {
            std::cerr << "Input file not specified. Use -i <model_file>\n";
            return false;
        }
        input_file_ = *iter;

        iter = get_cmd_option(argv, end, "-o");
        if (!iter)
        {
            std::cerr << "Output file not specified. Use -o <output_file>\n";
            return false;
        }
        if (++iter == end)
        {
            std::cerr << "Output file not specified. Use -o <output_file>\n";
            return false;
        }
        output_file_ = *iter;

        iter = get_cmd_option(argv, end, "-v");
        if (iter)
        {
            verbose_ = true;
        }

        return true;
    }

    const std::string& input_file() const { return input_file_; }
    const std::string& output_file() const { return output_file_; }
    bool verbose() const { return verbose_; };
};

class log_stream : public Assimp::LogStream
{
  public:
    // Write something using your own functionality
    void write(const char* message) { ::printf("%s", message); }
};

// convert assimp scene to model flatbuffer
class model_processor
{
    struct nodes_data
    {
        std::vector<content::Node> nodes;
        std::vector<content::Mat4> transforms;
        std::vector<std::string> names;
        std::vector<uint32_t> node_meshes;
        std::vector<uint32_t> node_children;
    };

  public:
    void process_node(const aiNode* node, nodes_data& data)
    {
        // create a node
        data.names.push_back(node->mName.C_Str());

        // add transform
        float m[16];
        for (int i = 0; i < 16; i++)
            m[i] = node->mTransformation[i / 4][i % 4];
        data.transforms.emplace_back(m);

        // add children
        std::vector<uint32_t> children_indices;
        for (uint32_t i = 0; i < node->mNumChildren; i++)
        {
            const aiNode* child = node->mChildren[i];
            process_node(child, data);
            children_indices.push_back(data.nodes.size());
        }
        uint32_t children_count = static_cast<uint32_t>(children_indices.size());
        uint32_t children_offset = children_count > 0 ? data.node_children.size() : 0;
        data.node_children.insert(data.node_children.end(), children_indices.begin(), children_indices.end());

        // add meshes
        std::vector<uint32_t> mesh_indices;
        for (uint32_t i = 0; i < node->mNumMeshes; i++)
            mesh_indices.push_back(node->mMeshes[i]);

        uint32_t mesh_count = static_cast<uint32_t>(mesh_indices.size());
        uint32_t mesh_offset = mesh_count > 0 ? data.node_meshes.size() : 0;
        data.node_meshes.insert(data.node_meshes.end(), mesh_indices.begin(), mesh_indices.end());

        // add node
        data.nodes.emplace_back(content::Node(mesh_offset, mesh_count, children_offset, children_count));
    }

    bool process(const aiScene* scene, const std::string& output_file)
    {
        flatbuffers::FlatBufferBuilder builder;

        content::ModelBuilder model_builder(builder);

        // compress meshes in a single vertex buffer for each channel
        std::vector<content::Vec3> positions;
        std::vector<content::Vec3> normals;
        std::vector<content::Vec3> tangents;
        std::vector<content::Vec3> bitangents;
        std::vector<content::Vec2> tex_coords;
        std::vector<uint32_t> indices;

        // meshes
        std::vector<content::Mesh> meshes;
        // materials

        uint32_t vertex_offset = 0;
        uint32_t index_offset = 0;

        // position vector
        for (uint32_t i = 0; i < scene->mNumMeshes; i++)
        {
            const aiMesh* mesh = scene->mMeshes[i];

            // positions
            positions.reserve(positions.size() + mesh->mNumVertices);
            for (uint32_t j = 0; j < mesh->mNumVertices; j++)
            {
                auto pos = mesh->mVertices[j];
                positions.push_back(content::Vec3(pos.x, pos.y, pos.z));
            }

            // normals
            if (mesh->HasNormals())
            {
                normals.reserve(normals.size() + mesh->mNumVertices);
                for (uint32_t j = 0; j < mesh->mNumVertices; j++)
                {
                    auto norm = mesh->mNormals[j];
                    normals.push_back(content::Vec3(norm.x, norm.y, norm.z));
                }
            }

            // tangents
            if (mesh->HasTangentsAndBitangents())
            {
                tangents.reserve(tangents.size() + mesh->mNumVertices);
                bitangents.reserve(bitangents.size() + mesh->mNumVertices);
                for (uint32_t j = 0; j < mesh->mNumVertices; j++)
                {
                    auto tan = mesh->mTangents[j];
                    tangents.push_back(content::Vec3(tan.x, tan.y, tan.z));

                    auto bitan = mesh->mBitangents[j];
                    bitangents.push_back(content::Vec3(bitan.x, bitan.y, bitan.z));
                }
            }

            // texture coordinates
            if (mesh->HasTextureCoords(0))
            {
                tex_coords.reserve(tex_coords.size() + mesh->mNumVertices);
                for (uint32_t j = 0; j < mesh->mNumVertices; j++)
                {
                    auto tex_coord = mesh->mTextureCoords[0][j];
                    tex_coords.push_back(content::Vec2(tex_coord.x, tex_coord.y));
                }
            }

            // indices
            uint32_t index_count = 0;
            indices.reserve(indices.size() + mesh->mNumFaces * 3);
            for (uint32_t j = 0; j < mesh->mNumFaces; j++)
            {
                const aiFace& face = mesh->mFaces[j];
                if (face.mNumIndices == 3)
                {
                    indices.push_back(face.mIndices[0]);
                    indices.push_back(face.mIndices[1]);
                    indices.push_back(face.mIndices[2]);
                    index_count += 3;
                }
            }

            // create mesh
            meshes.push_back(content::Mesh(vertex_offset, index_offset, index_count, mesh->mMaterialIndex));

            vertex_offset = positions.size();
            index_offset = indices.size();
        }

        // process materials
        std::vector<flatbuffers::Offset<content::Material>> materials;
        for (uint32_t i = 0; i < scene->mNumMaterials; i++)
        {
            const aiMaterial* material = scene->mMaterials[i];

            // create a material
            std::string material_name = "unknown_material" + std::to_string(i);
            aiString value;
            if (material->Get(AI_MATKEY_NAME, value) == AI_SUCCESS)
                material_name = value.C_Str();

            auto name = builder.CreateString(material_name);
            content::MaterialBuilder material_builder(builder);
            material_builder.add_name(name);
            materials.push_back(material_builder.Finish());

            // TODO: other material properties can be added here
        }

        // process nodes
        nodes_data nodes;
        process_node(scene->mRootNode, nodes);

        // create the model
        model_builder.add_positions(builder.CreateVectorOfStructs(positions));
        model_builder.add_normals(builder.CreateVectorOfStructs(normals));
        model_builder.add_tangents(builder.CreateVectorOfStructs(tangents));
        model_builder.add_binormals(builder.CreateVectorOfStructs(bitangents));
        model_builder.add_tex_coords(builder.CreateVectorOfStructs(tex_coords));
        model_builder.add_indices(builder.CreateVector(indices));
        model_builder.add_materials(builder.CreateVector(materials));
        model_builder.add_meshes(builder.CreateVectorOfStructs(meshes));
        model_builder.add_nodes(builder.CreateVectorOfStructs(nodes.nodes));
        model_builder.add_node_names(builder.CreateVectorOfStrings(nodes.names));
        model_builder.add_node_transforms(builder.CreateVectorOfStructs(nodes.transforms));
        model_builder.add_node_children(builder.CreateVector(nodes.node_children));
        model_builder.add_node_meshes(builder.CreateVector(nodes.node_meshes));
        auto model = model_builder.Finish();

        content::FinishModelBuffer(builder, model);

        // Save the flatbuffer to a file
        std::ofstream ofs(output_file, std::ios::binary);
        if (!ofs)
        {
            std::cerr << "Error creating output file: " << output_file << "\n";
            return false;
        }

        auto data = builder.GetBufferPointer();
        auto size = builder.GetSize();
        ofs.write(reinterpret_cast<const char*>(data), size);
        ofs.close();

        return true;
    }
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

    // create a logger instance
    Assimp::DefaultLogger::create("", Assimp::Logger::VERBOSE);

    uint32_t severity = Assimp::Logger::Err | Assimp::Logger::Warn;
    if (opts.verbose())
        severity |= Assimp::Logger::Debugging | Assimp::Logger::Info;

    // attaching it to the default logger
    auto stream = new log_stream();
    Assimp::DefaultLogger::get()->attachStream(stream, severity);

    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(opts.input_file().c_str(), aiProcessPreset_TargetRealtime_Quality);

    // kill it after the work is done
    Assimp::DefaultLogger::kill();

    if (!scene)
    {
        std::cerr << "Error loading model: " << importer.GetErrorString() << "\n";
        return EXIT_FAILURE;
    }

    model_processor processor;
    if (!processor.process(scene, opts.output_file()))
        return EXIT_FAILURE;

    // process the scene as needed
    std::cout << "Model processed successfully: " << opts.input_file() << "\n";

    return EXIT_SUCCESS;
}