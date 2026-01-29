#pragma once

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <fftw3.h>
#include <string.h>
#include <sndfile.h>

enum WindowFunction {
    WF_HANN,
};

typedef struct {
    enum WindowFunction window_function;

    size_t window_size;
    size_t hop_size;
    float* time_buf;
    fftwf_complex* freq_buf;

    fftwf_plan forward;
    fftwf_plan reverse;
} FFTKernel;

typedef struct {
    size_t sample_rate;

    size_t frames;
    float* data;

    int channels;
} Audiodata;

typedef struct {
    // There is no window function because this is not to be considered when doing OLA.

    // Justification: in a reversal operation, the user must explicitly want a change in sample rate.
    size_t sample_rate;

    // Justification: the original length of audio is rounded.
    size_t original_length;

    size_t window_count;

    // Guaranteed to have a size of sd->window_count * fk->window_size.
    // Yes, it is necessary associated with the kernel.
    fftwf_complex* data;

    // There is no option for interlacing windows. Just seems like unnecessary copying.
} Spectrodata;

Audiodata* audiodata_read_file(const char* fname)
{
    SF_INFO sfinfo = {};

    SNDFILE *sndfile = sf_open(fname, SFM_READ, &sfinfo);
    if (!sndfile) {
        fprintf(stderr, "Error opening audio file '%s': %s\n", fname, sf_strerror(NULL));
        return NULL;
    }

    Audiodata *ret = calloc(1, sizeof(Audiodata));
    assert(ret);

    ret->sample_rate = sfinfo.samplerate;
    ret->frames = sfinfo.frames;
    ret->channels = sfinfo.channels;

    ret->data = calloc(ret->frames * ret->channels, sizeof(float));
    assert(ret->data);
    
    (void) sf_readf_float(sndfile, ret->data, sfinfo.frames);

    sf_close(sndfile);
    return ret;
}

void audiodata_destroy(Audiodata* ad) {
    free(ad->data);
    free(ad);
}

FFTKernel* fftkernel_create(enum WindowFunction window_function, size_t window_size, size_t hop_size) {
    FFTKernel *ret = calloc(1, sizeof(FFTKernel));
    assert(ret);

    ret->window_function = window_function;

    ret->window_size = window_size;
    ret->hop_size = hop_size;
    ret->time_buf = fftwf_alloc_real(window_size);
    assert(ret->time_buf);
    ret->freq_buf = fftwf_alloc_complex(window_size / 2 + 1);
    assert(ret->freq_buf);

    ret->forward = fftwf_plan_dft_r2c_1d(window_size, ret->time_buf, ret->freq_buf, FFTW_PATIENT);
    assert(ret->forward);
    ret->reverse = fftwf_plan_dft_c2r_1d(window_size, ret->freq_buf, ret->time_buf, FFTW_PATIENT);
    assert(ret->reverse);
    
    return ret;
}

void fftkernel_destroy(FFTKernel* fk) {
    fftwf_destroy_plan(fk->forward);
    fftwf_destroy_plan(fk->reverse);
    fftwf_free(fk->time_buf);
    fftwf_free(fk->freq_buf);
    free(fk);
}

Spectrodata* fftkernel_execute_forward(const FFTKernel* fk, const Audiodata *ad) {
    if (ad->channels != 1) {
        fprintf(stderr, "fftkernel_execute_forward: An Audiodata that wasn't one channel was given. It had %d channels.", ad->channels);
        return NULL;
    }

    Spectrodata* sd = calloc(1, sizeof(Spectrodata));
    assert(sd);

    sd->sample_rate = ad->sample_rate;
    sd->original_length = ad->frames;

    sd->window_count = (ad->frames + fk->window_size - 1) / fk->window_size;
    sd->data = calloc((fk->window_size / 2 + 1) * sd->window_count, sizeof(fftwf_complex));
    assert(sd->data);

    fftwf_complex *sptr = sd->data;
    const float* aptr_end = ad->data + ad->frames;

    for (float* aptr = ad->data; aptr < aptr_end; aptr += fk->hop_size) {
        
        if (aptr + fk->hop_size > aptr_end) {
            memset(fk->time_buf, 0, fk->window_size);
            memcpy(fk->time_buf, aptr, aptr_end - aptr);
        } else {
            memcpy(fk->time_buf, aptr, fk->window_size * sizeof(float));
        }

        fftwf_execute(fk->forward);

        memcpy(sptr, fk->freq_buf, (fk->window_size / 2 + 1) * sizeof(fftwf_complex));

        sptr += (fk->window_size / 2 + 1);
    }
}

void spectrodata_destroy(Spectrodata *sd) {
    free(sd->data);
    free(sd);
}