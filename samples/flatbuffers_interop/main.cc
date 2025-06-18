#include <flatbuffers/flatbuffers.h>

#include "fbs/model_generated.h"
#include "fbs/test_generated.h"


std::string serialize()
{
    // Create a FlatBufferBuilder
    flatbuffers::FlatBufferBuilder builder;

    auto hero = hero::CreateWarrior(builder, builder.CreateString("Conan"), 100);
    builder.Finish(hero);

    return std::string((const char*)builder.GetBufferPointer(), builder.GetSize());
}

void deserialize(const std::string& data)
{
    // Verify the buffer
    if (!flatbuffers::Verifier((const uint8_t*)data.c_str(), data.size()).VerifyBuffer<hero::Warrior>())
    {
        printf("Error !\n");
        exit(EXIT_FAILURE);
    }
#
    // Get the hero from the buffer
    auto hero = hero::GetWarrior(data.c_str());

    // Print the hero's name and HP
    printf("Hero Name: %s, HP: %u\n", hero->name()->c_str(), hero->hp());
}

int main()
{
    // Serialize the hero
    auto data = serialize();

    printf("Serialized size: %zu bytes\n", data.size());

    // Deserialize the hero
    deserialize(data);

    return 0;
}