#include "fft.h"
#include <stdbool.h>

int main() {
    _app_config_create("test.wav", "out.png", 4096, 2048, SPEC_MAGNITUDE, true);
}