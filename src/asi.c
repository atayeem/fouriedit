#include "buffer.h"
#include <fftw3.h>
#include "fft.c"

typedef struct {
    Buffer_sample b;
    int channels;
} Audiodata;

// Ordering: This type is column-major. Every frame is interlaced.
// This makes it behave better with the other types, which are all
// column-major. Use Spectrodata_RM if this is not desirable.
typedef struct {
    Buffer_complex b;
    sample_count true_length;
    sample_count window_size;
    int window_count;
} Spectrodata;

typedef struct {
    Buffer_complex b;
    sample_count true_length;
    sample_count window_size;
    int window_count;
} Spectrodata_RM;

typedef struct {
    Buffer_byte b;
    int width;
    int height;
    int channels;
} Imagedata;

enum {
    ASI_1TO1x2C_BASIC,
    ASI_2TO1x4C_LR_COLORING,
    ASI_1TO1x4C_DOMAIN_COLORING,
    ASI_1TO2x2C_PHASE_AND_MAGNITUDE,
};

// The idea is that it should be always modified in-place, since this allows buffer reuse by default.

void audio_to_spectro(FFT_Context ctx, Audiodata* in, Spectrodata* out);
void spectro_to_audio(IFFT_Context ctx, Spectrodata* in, Audiodata* out);

// Generates a black-and-white (2ch) image with only the magnitude information displayed.
void spectro_to_image_basic(Spectrodata* in, Imagedata *out);

// Generates a colored (4ch) image where each input channel is assigned a color, and they are mixed together.
// the colors are in RGBA format. Pick two colors that add to white, you probably meant alpha to be 0xFF.
void spectro_to_image_lr_coloring(Spectrodata* left_in, Spectrodata* right_in, Imagedata *out, uint32_t left_color, uint32_t right_color);

// This one is similar to `basic`, but the hue of the color is based on the phase.
void spectro_to_image_domain_coloring(Spectrodata* in, Imagedata *out);

// This one generates two images, 
void spectro_to_image_phase_and_magnitude(Spectrodata* in, Imagedata* left_out, Imagedata* right_out);

void image_to_spectro_basic(Imagedata* in, Spectrodata* out);
void image_to_spectro_lr_coloring(Imagedata *in, Spectrodata* left_out, Spectrodata* right_out);
void image_to_spectro_domain_coloring(Imagedata *in, Spectrodata* out);
void image_to_spectro_phase_and_magnitude(Imagedata* left_in, Imagedata* right_in, Spectrodata* out);