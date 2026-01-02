#pragma once

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <fftw3.h>
#include <string.h>

enum WindowFunction {
    WF_HANN
};

typedef struct {
    size_t window_size;
    size_t hop_size;
    enum WindowFunction window_function;
} Specification;

typedef struct {
    const Specification* spec;

    float* time_buf;
    fftwf_complex* freq_buf;

    fftwf_plan forward;
    fftwf_plan reverse;
} FFTKernel;

typedef struct {
    size_t sample_count;
    const Specification* spec;

    const FFTKernel* fft;

    float *audio;
    float* spectrogram;
} JobContext;

FFTKernel* fft_kernel_create(const Specification* spec);
void fft_kernel_destroy(FFTKernel* fk);
void fft_kernel_execute(const FFTKernel* fk);

JobContext* job_context_create_forward(const Specification *spec, size_t num_samples, float *time_buf, fftwf_complex *freq_buf);
JobContext* job_context_create_reverse(const Specification *spec, size_t num_samples, fftwf_complex *time_buf, float *freq_buf);
void job_context_destroy(JobContext *jc);
JobContext* job_context_execute(const JobContext* jc);

FFTKernel* fft_kernel_create(const Specification *spec) {
    FFTKernel* fk = malloc(sizeof(FFTKernel));
    assert(fk);

    fk->spec = spec;

    fk->time_buf = fftwf_alloc_real(fk->spec->window_size);
    fk->freq_buf = fftwf_alloc_complex(fk->spec->window_size / 2 + 1);
    assert(fk->time_buf && fk->freq_buf);

    fk->forward = fftwf_plan_dft_r2c_1d(fk->spec->window_size, fk->time_buf, fk->freq_buf, FFTW_PATIENT);
    fk->reverse = fftwf_plan_dft_c2r_1d(fk->spec->window_size, fk->freq_buf, fk->time_buf, FFTW_PATIENT);

    return fk;
}

void fft_kernel_destroy(FFTKernel *fk) {
    fftwf_destroy_plan(fk->reverse);
    fftwf_destroy_plan(fk->forward);

    fftwf_free(fk->freq_buf);
    fftwf_free(fk->time_buf);

    free(fk);
}

void fft_kernel_execute_forward(const FFTKernel *fk, float *time_buf) {
}

static const Specification SP_HANN8192 = {
    .window_size = 8192,
    .hop_size = 4096,
    .window_function = WF_HANN,
};

