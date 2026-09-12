/*
 * Deterministic stand-in for the real native voiceprint library.
 * It implements the exact vp.h ABI so the Go binding can be built,
 * tested and demonstrated; swap in the real .a/.so in production.
 *
 * The model file genuinely parameterizes extraction: its bytes seed a
 * deterministic PRNG that expands into a projection matrix, and the
 * embedding is a model-dependent transform of the audio spectrum. The
 * same model file always yields the same embedding, while different
 * model files yield different embeddings for the same audio.
 */
#include "vp.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define VP_DIM 256
#define VP_MAX_ANALYSIS 8192
#define VP_BANDS 64 /* spectral bands feeding the model projection */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static _Thread_local char vp_tls_err[256];

static void vp_report(char *err, size_t cap, const char *msg) {
    if (err != NULL && cap > 0) {
        snprintf(err, cap, "%s", msg);
    }
    snprintf(vp_tls_err, sizeof(vp_tls_err), "%s", msg);
}

static void vp_clear(char *err, size_t cap) {
    if (err != NULL && cap > 0) {
        err[0] = '\0';
    }
    vp_tls_err[0] = '\0';
}

const char *vp_last_error(void) { return vp_tls_err; }

struct vp_engine {
    unsigned long model_size;
    unsigned long model_checksum;  /* FNV-1a of the model file */
    float proj[VP_DIM * VP_BANDS]; /* projection derived from the model */
};

/* splitmix64: a deterministic PRNG that expands the model identity into
 * projection weights, so the model bytes decide the feature transform. */
static uint64_t vp_next_u64(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* Next uniform variate in [-1, 1). */
static float vp_next_unit(uint64_t *state) {
    return (float)((double)(vp_next_u64(state) >> 40) / (double)(1ULL << 24) * 2.0 - 1.0);
}

vp_engine_t *vp_engine_create(const char *model_path, char *err, size_t err_cap) {
    if (model_path == NULL || model_path[0] == '\0') {
        vp_report(err, err_cap, "model path is empty");
        return NULL;
    }
    FILE *f = fopen(model_path, "rb");
    if (f == NULL) {
        vp_report(err, err_cap, "cannot open model file");
        return NULL;
    }
    unsigned long size = 0, hash = 14695981039346656037UL;
    unsigned char buf[4096];
    size_t r;
    while ((r = fread(buf, 1, sizeof(buf), f)) > 0) {
        for (size_t i = 0; i < r; i++) {
            hash = (hash ^ buf[i]) * 1099511628211UL;
        }
        size += (unsigned long)r;
    }
    fclose(f);
    if (size == 0) {
        vp_report(err, err_cap, "model file is empty");
        return NULL;
    }
    vp_engine_t *e = (vp_engine_t *)calloc(1, sizeof(*e));
    if (e == NULL) {
        vp_report(err, err_cap, "out of memory");
        return NULL;
    }
    e->model_size = size;
    e->model_checksum = hash;

    /* Expand the model identity into the projection applied at extract
     * time: the same model file always produces the same transform, and
     * different model files produce different embeddings. */
    uint64_t st = (uint64_t)hash ^ ((uint64_t)size * 0x9E3779B97F4A7C15ULL);
    double scale = 1.0 / sqrt((double)VP_BANDS);
    for (int i = 0; i < VP_DIM * VP_BANDS; i++) {
        e->proj[i] = (float)((double)vp_next_unit(&st) * scale);
    }
    vp_clear(err, err_cap);
    return e;
}

int vp_engine_dim(const vp_engine_t *e) {
    (void)e;
    return VP_DIM;
}

int vp_engine_extract(vp_engine_t *e,
                      const float *samples, int n, int sample_rate,
                      float *out, int cap,
                      vp_progress_fn cb, void *user,
                      char *err, size_t err_cap) {
    if (e == NULL) {
        vp_report(err, err_cap, "engine is NULL");
        return -1;
    }
    if (samples == NULL || n <= 0) {
        vp_report(err, err_cap, "no input samples");
        return -1;
    }
    if (sample_rate < 8000 || sample_rate > 48000) {
        vp_report(err, err_cap, "unsupported sample rate (want 8000..48000)");
        return -1;
    }
    if (out == NULL || cap < VP_DIM) {
        vp_report(err, err_cap, "output buffer too small");
        return -1;
    }

    int m = n < VP_MAX_ANALYSIS ? n : VP_MAX_ANALYSIS;

    /* Front end (model-independent): log-magnitude of a coarse DFT,
     * reduced to VP_BANDS spectral bands. */
    float bands[VP_BANDS];
    for (int b = 0; b < VP_BANDS; b++) {
        double freq = (double)(b + 1) * ((double)sample_rate / 2.0) / (double)VP_BANDS;
        double w = 2.0 * M_PI * freq / (double)sample_rate;
        double re = 0.0, im = 0.0;
        for (int i = 0; i < m; i++) {
            re += (double)samples[i] * cos(w * (double)i);
            im -= (double)samples[i] * sin(w * (double)i);
        }
        bands[b] = (float)log1p(sqrt(re * re + im * im) / (double)m);
    }

    /* Model-dependent transform: the projection derived from the model
     * file turns the spectrum into the embedding. */
    for (int k = 0; k < VP_DIM; k++) {
        const float *row = e->proj + (size_t)k * VP_BANDS;
        double s = 0.0;
        for (int b = 0; b < VP_BANDS; b++) {
            s += (double)row[b] * (double)bands[b];
        }
        out[k] = (float)tanh(s);
        if (cb != NULL && ((k & 63) == 63 || k == VP_DIM - 1)) {
            cb((double)(k + 1) / (double)VP_DIM, user);
        }
    }
    vp_clear(err, err_cap);
    return 0;
}

float vp_cosine(const float *a, const float *b, int dim) {
    if (a == NULL || b == NULL || dim <= 0) {
        vp_report(NULL, 0, "invalid vectors");
        return 0.0f;
    }
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < dim; i++) {
        dot += (double)a[i] * (double)b[i];
        na += (double)a[i] * (double)a[i];
        nb += (double)b[i] * (double)b[i];
    }
    if (na == 0.0 || nb == 0.0) {
        vp_report(NULL, 0, "zero-norm vector");
        return 0.0f;
    }
    return (float)(dot / (sqrt(na) * sqrt(nb)));
}

void vp_engine_destroy(vp_engine_t *e) { free(e); }
