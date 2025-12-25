#pragma once

#include <stddef.h>
#include <stdint.h>

#include <fftw3.h>

enum SpectrogramType {
    SPEC_MAGNITUDE,
};

struct AppConfig {
    const char *input_path;
    const char *output_path;

    size_t sample_rate;
    size_t window_size;
    size_t hop_size;

    enum SpectrogramType type;
    int reverse;
};

struct FFTcontext {
    size_t window_size;
    size_t hop_size;

    float* time_buf;
    fftwf_complex* freq_buf;

    fftwf_plan forward;
    fftwf_plan inverse;
};

struct JobContext {
    /* audio domain */
    size_t sample_rate;
    size_t channels;
    size_t num_samples;

    float *audio;

    size_t width;
    size_t height;

    float* spectrogram;

    struct fft_context *fft;
};
