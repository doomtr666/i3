struct vertex_in
{
    float4 position : POSITION;
    float3 normal : NORMAL;
};


struct vertex_out
{
    float4 position : SV_Position;
    float3 normal : NORMAL;
};

[shader("vertex")]
vertex_out vertex_main(vertex_in input)
{
    vertex_out output;
    output.position = input.position; // Assuming modelViewProjection is applied elsewhere
    output.normal = input.normal; // Pass through normal
    return output;
}

struct fragment_out
{
    float4 color : SV_Target0;
    float4 normal: SV_Target1;
    float4 metalic_roughness: SV_Target2;
};

[shader("fragment")]
fragment_out fragment_main(vertex_out input)
{
    fragment_out output;
    output.color = float4(1, 0, 0, 1); // Red color
    output.normal = float4(0, 0, 1, 1); // Normal pointing in Z direction
    output.metalic_roughness = float4(0, 0, 0, 1); // No metallic or roughness
    return output;
}
