#include "virtual_alloc.h"

#if I3_PLATFORM == I3_PLATFORM_WINDOWS

    #include <Windows.h>

void* i3_virtual_alloc(size_t size)
{
    return VirtualAlloc(NULL, size, MEM_RESERVE, PAGE_NOACCESS);
}

bool i3_virtual_free(void* address)
{
    return VirtualFree(address, 0, MEM_RELEASE) == TRUE;
}

bool i3_virtual_commit(void* address, size_t size)
{
    return VirtualAlloc(address, size, MEM_COMMIT, PAGE_READWRITE) != NULL;
}

bool i3_virtual_decommit(void* address, size_t size)
{
    return VirtualFree(address, size, MEM_DECOMMIT) == TRUE;
}

#endif
