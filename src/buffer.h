#pragma once

#include <stdlib.h>
#include <fftw3.h>
#include "typename.h"

#define NDEBUG
#include <assert.h>

#define CREATE_BUFFER_TYPE(T) \
\
typedef struct { \
    T* data;\
    size_t size; \
    size_t capacity; \
} Buffer_##T; \
\
static Buffer_##T create_buffer_##T(size_t size) { \
    assert(size > 0);\
    Buffer_##T out; \
    out.data = (T*) malloc(sizeof(T) * size); \
    out.size = size; \
    out.capacity = size; \
    assert(out.data);\
    return out; \
} \
\
static void destroy_buffer_##T(Buffer_##T* b) {\
    free(b->data); \
    b->size = 0;\
    b->capacity = 0; \
} \
\
static Buffer_##T create_buffer_##T##_from_pointer(T* ptr, size_t size) { \
    return (Buffer_##T) {ptr, size};\
}

CREATE_BUFFER_TYPE(sample);
CREATE_BUFFER_TYPE(complex);
CREATE_BUFFER_TYPE(byte);

#undef NDEBUG