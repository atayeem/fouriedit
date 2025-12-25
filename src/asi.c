#include "buffer.h"
#include <stdbool.h>
#include <sndfile.h>
#include <fftw3.h>
#include "fft.c"

#define FATAL "[FATAL] "
#define ERROR "[ERROR] "
#define WARN "[WARN] "
#define INFO "[INFO] "
#define LOG printf

typedef struct {
    Buffer_sample b;
    SF_INFO i;
} Audiodata;

// Ordering: This type is column-major. Every frame is interlaced.
// This makes it behave better with the other types, which are all
// column-major.
typedef struct {
    Buffer_complex b;
    sample_count true_length;
    sample_count window_size;
    int window_count;
} Spectrodata;

typedef struct {
    Buffer_byte b;
    int width;
    int height;
    int channels;
} Imagedata;

// The file I/O section!

// Open an audio file and read the channels.
bool read_audio(const char* fname, Audiodata* ad) {
    SF_INFO sfinfo;
    SNDFILE* s = sf_open(fname, SFM_READ, &sfinfo);
    if (!s) {
        LOG("read_audio: failed to open file %s\n", fname);
        return false;
    }

    ad->i = sfinfo;
    buf_resize_soft_sample(&ad->b, sfinfo.frames * sfinfo.channels);
    sf_count_t written = sf_readf_double(s, ad->b.data, sfinfo.frames);

    if (written < sfinfo.frames)
        LOG(WARN "read_audio: Could not read entire file.");
    return true;
}


// The idea is that it should be always modified in-place, since this allows buffer reuse by default.
// They do not return error codes because they do operations that can't fail; they don't do syscalls.
// In the future maybe they should modify errno?

// Turns *mono* audio into a spectrum. Requires a prepared FFT_Context.
void audio_to_spectro(FFT_Context ctx, Audiodata* in, Spectrodata* out);

// Turns a spectrogram into *mono* audio.
void spectro_to_audio(IFFT_Context ctx, Spectrodata* in, Audiodata* out);

// Generates a black-and-white (2ch) image with only the magnitude information displayed.
void spectro_to_image_basic(Spectrodata* in, Imagedata *out);

// Generates a colored (4ch) image where each input channel is assigned a color, and they are mixed together.
// the colors are in RGBA format. Pick two colors that add to white, you probably meant alpha to be 0xFF.
void spectro_to_image_lr_coloring(Spectrodata* left_in, Spectrodata* right_in, Imagedata *out, uint32_t left_color, uint32_t right_color);

// This one is similar to `basic`, but the hue of the color is based on the phase.
void spectro_to_image_domain_coloring(Spectrodata* in, Imagedata *out);

// This one generates two images, both are greyscale (2ch) representations of the phase and magnitude respectively.
void spectro_to_image_phase_and_magnitude(Spectrodata* in, Imagedata* left_out, Imagedata* right_out);

// Turns a black-and-white image (2ch) into a spectrum. This will sound really weird if you are trying to do this
// from normal audio! Maybe that's what you want!
void image_to_spectro_basic(Imagedata* in, Spectrodata* out);

// Turns a colored image into two spectra, splitting it into two magnitude-only ones. Also loses phase information!
void image_to_spectro_lr_coloring(Imagedata *in, Spectrodata* left_out, Spectrodata* right_out, uint32_t left_color, uint32_t right_color);

// Turns a domain-colored spectrogram into a spectrum.
void image_to_spectro_domain_coloring(Imagedata *in, Spectrodata* out);

// Turns two images, one encoding phase, one encoding magnitude, into a spectrum.
void image_to_spectro_phase_and_magnitude(Imagedata* left_in, Imagedata* right_in, Spectrodata* out);