#include <stdio.h>

#include "add.h"

#include <vulkan/vulkan.h>

int main()
{
    int a = 5;
    int b = 10;
    int result = add(a, b);
    printf("The sum of %d and %d is %d\n", a, b, result);

    VkInstance instance;
    VkInstanceCreateInfo createInfo = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    };

    VkResult res = vkCreateInstance(&createInfo, NULL, &instance);
    if (res != VK_SUCCESS) {
        printf("Failed to create Vulkan instance: %d\n", res);
        return -1;
    }

    printf("Vulkan instance created successfully!\n");

    vkDestroyInstance(instance, NULL);


    return 0;
}