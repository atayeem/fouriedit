#include <stdlib.h>

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
static Buffer_##T* create_buffer_##T(size_t size) { \
    \
}

CREATE_BUFFER_TYPE(int)

#undef NDEBUG