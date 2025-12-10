#include <stdio.h>
#include <stdlib.h>
#include <fftw3.h>
#include "buffer.h"

#define NDEBUG
#include <assert.h>

typedef struct {
    sample*      in;
    complex*     out;
    sample_count window;
    fftw_plan    plan;
} FFT_Context;

typedef struct {
    complex*     in;
    sample*      out;
    sample_count window;
    fftw_plan    plan;
} IFFT_Context;

FFT_Context create_fft_context(size_t window);
void destroy_fft_context(FFT_Context* fc);
void execute_fft(FFT_Context* fc);

IFFT_Context create_ifft_context(size_t window);
void destroy_ifft_context(IFFT_Context* fc);
void execute_ifft(IFFT_Context* fc);

FFT_Context create_fft_context(size_t window) {
    assert(window > 0);
    assert(window < (1 << 20));

    FFT_Context fc;

    fc.in = fftw_alloc_real(window);
    fc.out = fftw_alloc_complex(window/2 + 1);
    assert(fc.in && fc.out);

    fc.window = window;
    fc.plan = fftw_plan_dft_r2c_1d(window, fc.in, fc.out, FFTW_PATIENT);
    assert(fc->plan);

    return fc;
}

void destroy_fft_context(FFT_Context* fc) {
    fftw_destroy_plan(fc->plan);
    fftw_free(fc->in);
    fftw_free(fc->out);
}

void execute_fft(FFT_Context* fc) {
    
}

IFFT_Context create_ifft_context(size_t window) {
    assert(window > 0);
    assert(window < (1 << 20));

    IFFT_Context ic;

    ic.in = fftw_alloc_complex(window/2 + 1);
    ic.out = fftw_alloc_real(window);
    assert(ic->in && ic->out);

    ic.window = window;
    ic.plan = fftw_plan_dft_c2r_1d(window, ic.in, ic.out, FFTW_PATIENT);
    assert(ic->plan);

    return ic;
}

void destroy_ifft_context(IFFT_Context* ic) {
    fftw_destroy_plan(ic->plan);
    fftw_free(ic->in);
    fftw_free(ic->out);
}