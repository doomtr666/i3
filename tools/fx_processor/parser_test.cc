#include <gtest/gtest.h>

#include "parser.h"

class ParserTest : public testing::Test
{
    parser parser_;

  public:
    bool parser(const char* src) { return parser_.parse(src) != nullptr; }
};

TEST_F(ParserTest, empty)
{
    EXPECT_TRUE(parser(""));
}

TEST_F(ParserTest, invalid)
{
    EXPECT_FALSE(parser("$"));
}

TEST_F(ParserTest, commments)
{
    EXPECT_TRUE(parser("  // comment"));
    EXPECT_TRUE(parser(" /* comment */  "));
    EXPECT_TRUE(parser("/* comment\n *  / comment */"));
    EXPECT_FALSE(parser("/* comment"));
}

TEST_F(ParserTest, import)
{
    EXPECT_TRUE(parser("  import  \"test.i3fx\"  ;"));
    EXPECT_TRUE(parser("import \"test.i3fx\";\nimport \"test2.i3fx\";"));
    EXPECT_FALSE(parser("import \"test.i3fx\""));
}

TEST_F(ParserTest, parameters)
{
    const char* src = R"x(

    // simple parameter
    float2 p1;

    // parameter with annotation
    [default(float2(0,0))]
    float2 p2;

    // parameter with annotations
    [default(float2(0,0))]
    [min(float2(0,0))]
    [max(float2(1,1))]
    float2 p3;

    )x";

    EXPECT_TRUE(parser(src));
}

TEST_F(ParserTest, slang_section)
{
    const char* src = R"x(
    #slang

[shader("vertex")]
VertexStageOutput vertexMain(
    AssembledVertex assembledVertex)
{
    VertexStageOutput output;

    float3 position = assembledVertex.position;
    float3 normal   = assembledVertex.normal;

    output.coarseVertex.normal = mul(world,float4(normal,0)).xyz;
    output.coarseVertex.tangent = mul(world,float4(assembledVertex.tangent,0)).xyz;
    output.coarseVertex.binormal = mul(world,float4(assembledVertex.binormal,0)).xyz;
    output.coarseVertex.uv = assembledVertex.uv;

    output.sv_position = mul(mul(projView, world), float4(position, 1.0));

    return output;
}

    #end
    )x";

    EXPECT_TRUE(parser(src));
}

TEST_F(ParserTest, pipeline)
{
    const char* src = R"x(

    // empty pipeline
    compute empty {
    }

    // pipeline with states
    graphics with_states {

        shaders = 
        [
            { type = vertex_shader, entry = "vertexMain" },
            { type = fragment_shader, entry = "fragmentMain" }
        ]
    }

    )x";

    EXPECT_TRUE(parser(src));
}
