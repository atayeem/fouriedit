#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <fftw3.h>
#include <string.h>
#include <sndfile.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))

static float* generate_hann_window(size_t sz) {
    float* w = calloc(sz, sizeof(float));
    assert(w);

    for (size_t n = 0; n < sz; n++) {
        w[n] = 0.5 * (1 - cos(2.0 * M_PI * n / (sz - 1)));
    }
    return w;
}

static float* generate_none_window(size_t sz) {
    float* w = calloc(sz, sizeof(float));
    assert(w);

    for (size_t n = 0; n < sz; n++) {
        w[n] = 1.0;
    }
    return w;
}

enum WindowFunction {
    WF_HANN,
    WF_NONE,
};

typedef struct {
    float* window_function;

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

// Remember, this stores them as a contiguous array of Audiodata, not an array of pointers to Audiodata.
typedef struct {
    int count;
    Audiodata* data;
} AudiodataMany;

typedef struct {
    // Justification: in a reversal operation, the default behavior (non-specified) should be to preserve sample rate.
    size_t sample_rate;

    // Justification: the original length of audio is rounded.
    size_t original_length;

    size_t window_count;

    // Guaranteed to have a size of sd->window_count * (fk->window_size / 2 + 1).
    // Yes, it is necessarily associated with the kernel.
    fftwf_complex* data;

    // There is no option for interlacing windows. Just seems like unnecessary copying.
} Spectrodata;

AudiodataMany* audiodata_split_channels(const Audiodata* ad) {
    AudiodataMany* am = calloc(1, sizeof(AudiodataMany));
    assert(am);

    int channel_count = ad->channels;
    am->count = channel_count;
    am->data = calloc(channel_count, sizeof(Audiodata));
    assert(am->data);

    for (int i = 0; i < channel_count; i++) {
        Audiodata new_ad = {
            .channels = 1,
            .data = calloc(ad->frames, sizeof(float)),
            .frames = ad->frames,
            .sample_rate = ad->sample_rate
        };
        assert(new_ad.data);
        am->data[i] = new_ad;
    }

    for (size_t frame = 0; frame < ad->frames; frame++) {
        for (int channel = 0; channel < channel_count; channel++) {
            am->data[channel].data[frame] = ad->data[channel_count * frame + channel];
        }
    }

    return am;
}

void audiodata_many_destroy(AudiodataMany *am) {
    for (int i = 0; i < am->count; i++)
        free(am->data[i].data);
    free(am->data);
    free(am);
}

// Check return value.
Audiodata* audiodata_read_file(const char* fname) {
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

void audiodata_write_file(const char* fname, const Audiodata* ad) {
    SF_INFO sfinfo = {
        .channels = ad->channels,
        .format = SF_FORMAT_WAV | SF_FORMAT_PCM_16,
        .frames = ad->frames,
        .samplerate = ad->sample_rate,
    };

    SNDFILE *sndfile = sf_open(fname, SFM_WRITE, &sfinfo);
    if (!sndfile) {
        fprintf(stderr, "Failed to open audio file for writing '%s': %s\n", fname, sf_strerror(NULL));
        return;
    }

    sf_count_t written = sf_writef_float(sndfile, ad->data, ad->frames);
    if ((size_t)written < ad->frames) {
        fprintf(stderr, "Couldn't write all frames (%lld/%lld) to audio file '%s': %s\n", written, ad->frames, fname, sf_strerror(sndfile));
    }
    fprintf(stderr, "Yeah. %s\n", sf_strerror(sndfile));

    sf_close(sndfile);
}

void audiodata_destroy(Audiodata* ad) {
    free(ad->data);
    free(ad);
}

// Must succeed.
FFTKernel* fftkernel_create(enum WindowFunction window_function, size_t window_size, size_t hop_size) {
    FFTKernel *ret = calloc(1, sizeof(FFTKernel));
    assert(ret);

    switch(window_function) {
        case WF_NONE:
            ret->window_function = generate_none_window(window_size);
        break;

        case WF_HANN:
            ret->window_function = generate_hann_window(window_size);
        break;
    }

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
    free(fk->window_function);
    free(fk);
}

Spectrodata* fftkernel_execute_forward(const FFTKernel* fk, const Audiodata* ad) {
    if (ad->channels != 1) {
        fprintf(stderr, "fftkernel_execute_forward: An Audiodata that wasn't one channel was given. It had %d channels.\n", ad->channels);
        return NULL;
    }

    Spectrodata *const sd = calloc(1, sizeof(Spectrodata));
    assert(sd);

    sd->sample_rate = ad->sample_rate;
    sd->original_length = ad->frames;

    sd->window_count = (ad->frames + fk->window_size - 1) / fk->window_size;
    sd->data = calloc((fk->window_size / 2 + 1) * sd->window_count, sizeof(fftwf_complex));
    assert(sd->data);

    fftwf_complex *sptr = sd->data;
    const float* const aptr_end = ad->data + ad->frames;

    for (const float* aptr = ad->data; aptr < aptr_end; aptr += fk->hop_size) {
        
        if (aptr + fk->hop_size > aptr_end) {
            memset(fk->time_buf, 0, fk->window_size);
            memcpy(fk->time_buf, aptr, aptr_end - aptr);
        } else {
            memcpy(fk->time_buf, aptr, fk->window_size * sizeof(float));
        }

        // Hanning or whatever else
        for (size_t i = 0; i < fk->window_size; i++) {
            fk->time_buf[i] *= fk->window_function[i];
        }

        fftwf_execute(fk->forward);

        memcpy(sptr, fk->freq_buf, (fk->window_size / 2 + 1) * sizeof(fftwf_complex));

        sptr += (fk->window_size / 2 + 1);
    }

    return sd;
}

Audiodata* fftkernel_execute_reverse(const FFTKernel* fk, const Spectrodata* sd) {
    Audiodata *const ad = calloc(1, sizeof(Audiodata));
    assert(ad);
    ad->data = calloc(sd->original_length, sizeof(float));
    assert(ad->data);
    ad->channels = 1;
    ad->frames = sd->original_length;
    ad->sample_rate = sd->sample_rate;
    
    const size_t spec_size = fk->window_size / 2 + 1;
    const fftwf_complex *const sptr_end = sd->data + sd->window_count * spec_size;

    float* aptr = ad->data;
    const float* const aptr_end = ad->data + ad->frames;
    for (const fftwf_complex* sptr = sd->data; sptr < sptr_end; sptr += spec_size) {
        // No secondary check required because it's going to be a perfect multiple of the block size for spectrodata.

        memcpy(fk->freq_buf, sptr, spec_size);
        
        fftwf_execute(fk->reverse);

        // OLA algorithm
        for (size_t i = 0; i < MIN(fk->window_size, (size_t)(aptr_end - aptr)); i++)
            aptr[i] += fk->time_buf[i];
        
        aptr += fk->hop_size;
    }

    return ad;
}

void spectrodata_destroy(Spectrodata *sd) {
    free(sd->data);
    free(sd);
}

int main() {
    printf("Hello world!");
    FFTKernel *fk = fftkernel_create(WF_HANN, 4096, 2048);

    Audiodata *ad = audiodata_read_file("bin\\test.wav");
    assert(ad);

    AudiodataMany *am = audiodata_split_channels(ad);

    Spectrodata *sd = fftkernel_execute_forward(fk, &am->data[0]);
    assert(sd);

    Audiodata *ad2 = fftkernel_execute_reverse(fk, sd);
    assert(ad2);

    printf("Got here.\n");
    audiodata_write_file("bin\\test_out.wav", ad2);

    audiodata_destroy(ad);
    audiodata_many_destroy(am);
    fftkernel_destroy(fk);
    spectrodata_destroy(sd);

    return 0;
}