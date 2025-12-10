#pragma once

#include <stdlib.h>
#include <fftw3.h>
#include "typename.h"
#include <assert.h>

#define CREATE_BUFFER_TYPE(T) \
\
typedef struct { \
    T* data;\
    size_t size; \
    size_t capacity; \
} Buffer_##T; \
\
static Buffer_##T buf_create_##T(size_t size) { \
    assert(size > 0);\
    Buffer_##T out; \
    out.data = (T*) malloc(sizeof(T) * size); \
    out.size = size; \
    out.capacity = size; \
    assert(out.data); \
    return out; \
} \
\
static void buf_destroy_##T(Buffer_##T* b) {\
    free(b->data); \
    b->size = 0;\
    b->capacity = 0; \
} \
\
static Buffer_##T buf_create_from_pointer_##T(T* ptr, size_t size) { \
    return (Buffer_##T) {ptr, size};\
} \
\
static void buf_resize_hard_##T(Buffer_##T* b, size_t new_size) { \
    assert(new_size); \
    b->data = realloc(b->data, new_size); \
    assert(b->data); \
    b->size = new_size; \
    b->capacity = new_size; \
} \
\
static void buf_resize_soft_##T(Buffer_##T* b, size_t new_size) { \
    assert(new_size); \
    if (new_size <= b->capacity) { \
        b->size = new_size; \
    } else { \
        buf_resize_hard_##T(b, new_size); \
    } \
} \
\
static void buf_copy_##T(Buffer_##T* dst, Buffer_##T* src) { \
    resize_buffer_##T##_soft(dst, src->size); \
    memcpy(dst->data, src->data, src->size); \
} \

CREATE_BUFFER_TYPE(sample);
CREATE_BUFFER_TYPE(complex);
CREATE_BUFFER_TYPE(byte);