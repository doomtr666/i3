#pragma once

#include "common.h"

I3_EXPORT void* i3_virtual_alloc(size_t size);
I3_EXPORT bool i3_virtual_free(void* address);

I3_EXPORT bool i3_virtual_commit(void* address, size_t size);
I3_EXPORT bool i3_virtual_decommit(void* address, size_t size);
