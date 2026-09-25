/*
 * voiceprint.c - reference implementation of the voiceprint C ABI.
 *
 * This is a self-contained, deterministic DSP reference (no external
 * dependencies): each embedding holds frame-level loudness/voicing
 * moments followed by a mean log-frequency spectral-envelope profile;
 * embeddings are compared with cosine similarity. Swap this file for a
 * real speaker-recognition engine while keeping the ABI unchanged - the
 * Go binding does the rest.
 *
 * All host-provided buffers are copied. Nothing host-owned is retained
 * after a call returns.
 */
#include "voiceprint.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct sp_model {
    uint32_t magic;
    uint32_t version;
    uint32_t sample_rate;
    uint32_t dim;
};

#define MIN_SAMPLE_RATE 8000u
#define MAX_SAMPLE_RATE 192000u
#define MIN_DIM         12u
#define MAX_DIM         1024u

/* 50 ms minimum speech, 25 ms frames, 50 % overlap. */
#define MIN_SECONDS     0.05
#define FRAME_SECONDS   0.025
#define HOP_SECONDS     0.0125
#define SILENCE_LEVEL   1e-5f

typedef struct {
    float  *data;
    size_t  cap;
    size_t  len;
} fbuf;

static sp_status_t fbuf_push(fbuf *b, float v)
{
    if (b->len == b->cap) {
        size_t ncap = b->cap ? b->cap * 2 : 4096;
        float *nd = (float *)realloc(b->data, ncap * sizeof(float));
        if (!nd) return SP_ERR_NOMEM;
        b->data = nd;
        b->cap  = ncap;
    }
    b->data[b->len++] = v;
    return SP_OK;
}

const char *sp_strerror(sp_status_t code)
{
    static const char *const msgs[] = {
        "success",
        "invalid argument",
        "out of memory",
        "input/output error",
        "invalid or corrupt model file",
        "audio too short: at least 50ms of PCM is required",
        "audio is silence: not enough acoustic energy",
        "embedding dimension mismatch",
    };
    unsigned i = (unsigned)code;
    if (i <= SP_ERR_DIM_MISMATCH)
        return msgs[i];
    return "unknown error";
}

static sp_status_t model_check(const sp_model_t *m)
{
    if (!m || m->magic != SP_MAGIC || m->version != SP_VERSION ||
        m->sample_rate < MIN_SAMPLE_RATE ||
        m->sample_rate > MAX_SAMPLE_RATE ||
        m->dim < MIN_DIM || m->dim > MAX_DIM)
        return SP_ERR_BAD_MODEL;
    return SP_OK;
}

static sp_status_t read_model_bytes(FILE *fp, sp_model_t **out)
{
    sp_model_t *m = (sp_model_t *)calloc(1, sizeof(*m));
    if (!m) return SP_ERR_NOMEM;
    if (fread(&m->magic, sizeof(m->magic), 1, fp) != 1 ||
        fread(&m->version, sizeof(m->version), 1, fp) != 1 ||
        fread(&m->sample_rate, sizeof(m->sample_rate), 1, fp) != 1 ||
        fread(&m->dim, sizeof(m->dim), 1, fp) != 1) {
        free(m);
        return SP_ERR_IO;
    }
    if (model_check(m) != SP_OK) {
        free(m);
        return SP_ERR_BAD_MODEL;
    }
    *out = m;
    return SP_OK;
}

sp_status_t sp_model_open_file(const char *path, sp_model_t **out)
{
    if (!path || !out) return SP_ERR_ARGUMENT;
    FILE *fp = fopen(path, "rb");
    if (!fp) return SP_ERR_IO;
    sp_status_t st = read_model_bytes(fp, out);
    fclose(fp);
    return st;
}

sp_status_t sp_write_default_model(const char *path)
{
    if (!path) return SP_ERR_ARGUMENT;
    FILE *fp = fopen(path, "wb");
    if (!fp) return SP_ERR_IO;
    uint32_t hdr[4] = { SP_MAGIC, SP_VERSION, 16000u, 48u };
    size_t w = fwrite(hdr, sizeof(uint32_t), 4, fp);
    if (fclose(fp) != 0) return SP_ERR_IO;
    return w == 4 ? SP_OK : SP_ERR_IO;
}

sp_status_t sp_model_open_default(sp_model_t **out)
{
    if (!out) return SP_ERR_ARGUMENT;
    sp_model_t *m = (sp_model_t *)calloc(1, sizeof(*m));
    if (!m) return SP_ERR_NOMEM;
    m->magic       = SP_MAGIC;
    m->version     = SP_VERSION;
    m->sample_rate = 16000u;
    m->dim         = 48u;
    *out = m;
    return SP_OK;
}

void sp_model_close(sp_model_t *m)
{
    /* No embedded scratch state yet; idempotent. */
    free(m);
}

uint32_t sp_model_sample_rate(const sp_model_t *m)
{
    return (m && model_check(m) == SP_OK) ? m->sample_rate : 0;
}

uint32_t sp_model_dim(const sp_model_t *m)
{
    return (m && model_check(m) == SP_OK) ? m->dim : 0;
}

void sp_embedding_free(sp_embedding_t *e)
{
    if (!e) return;
    free(e->data);
    free(e);
}

/* Core DSP. samples are copied by the caller; this function only reads.
 *
 * Four per-frame descriptors are pooled into L2-normalised histograms:
 *   rms      - frame energy distribution (captures amplitude modulation)
 *   zcr      - zero-crossing rate (voiced/unvoiced mix)
 *   centroid - spectral centroid from a small DFT (brightness; tracks f0)
 *   pitch    - fraction of spectral energy in the 70..400 Hz pitch band
 */
static sp_status_t compute_embedding(const sp_model_t *m,
                                     const float *samples, size_t n,
                                     sp_embedding_t **out)
{
    if (n < (size_t)(m->sample_rate * MIN_SECONDS))
        return SP_ERR_TOO_SHORT;

    size_t flen = (size_t)(m->sample_rate * FRAME_SECONDS);
    size_t hop  = (size_t)(m->sample_rate * HOP_SECONDS);
    size_t nfr  = n > flen ? 1 + (n - flen) / hop : 0;
    if (nfr < 2 || flen < 2)
        return SP_ERR_TOO_SHORT;

    float *rms = (float *)malloc(nfr * sizeof(float));
    float *zcr = (float *)malloc(nfr * sizeof(float));
    if (!rms || !zcr) {
        free(rms); free(zcr);
        return SP_ERR_NOMEM;
    }

    unsigned dim = m->dim;
    unsigned nbands = dim - 4;          /* 4 moments + one log-band profile */
    double *band = (double *)calloc(nbands, sizeof(double));
    if (!band) {
        free(rms); free(zcr);
        return SP_ERR_NOMEM;
    }

    /* Spectral descriptor: fixed 512-point Hann-windowed DFT centered on
     * each frame; cost and resolution stay bounded at any sample rate. */
    const size_t nfft = 512;
    float *sintab = (float *)malloc(nfft * sizeof(float));
    float *costab = (float *)malloc(nfft * sizeof(float));
    float *window = (float *)malloc(nfft * sizeof(float));
    double *blow = (double *)malloc((nbands + 1) * sizeof(double));
    double *bcenter = (double *)malloc(nbands * sizeof(double));
    if (!sintab || !costab || !window || !blow || !bcenter) {
        free(rms); free(zcr); free(band);
        free(sintab); free(costab); free(window); free(blow); free(bcenter);
        return SP_ERR_NOMEM;
    }
    const double tau = 2.0 * 3.14159265358979323846;
    for (size_t i = 0; i < nfft; i++) {
        sintab[i] = (float)sin(tau * i / (double)nfft);
        costab[i] = (float)cos(tau * i / (double)nfft);
        window[i] = (float)(0.5 - 0.5 * cos(tau * (double)i /
                                           (double)(nfft - 1)));
    }

    /* Logarithmic band edges from 60 Hz to half the Nyquist frequency. */
    double f_lo = 60.0;
    double f_hi = (double)m->sample_rate * 0.25;
    double binHz = (double)m->sample_rate / (double)nfft;
    double lr = log(f_hi / f_lo) / (double)nbands;
    for (unsigned b = 0; b <= nbands; b++)
        blow[b] = f_lo * exp(lr * b) / binHz;
    for (unsigned b = 0; b < nbands; b++)
        bcenter[b] = 0.5 * (blow[b] + blow[b + 1]);

    size_t nbin = nfft / 2;
    int any_signal = 0;
    double voiced_frames = 0.0;

    for (size_t f = 0; f < nfr; f++) {
        size_t start = f * hop;

        /* ---- time domain: RMS + zero crossings ---- */
        double energy = 0.0, zc = 0.0;
        int prev = samples[start] >= 0.0f;
        for (size_t i = 0; i < flen; i++) {
            float sv = samples[start + i];
            energy += (double)sv * sv;
            int cur = sv >= 0.0f;
            if (i > 0 && cur != prev) zc += 1.0;
            prev = cur;
        }
        rms[f] = (float)sqrt(energy / flen);
        zcr[f] = (float)(zc / (flen - 1));
        if (rms[f] > SILENCE_LEVEL) any_signal = 1;

        /* ---- frequency domain: accumulate the log-band profile ---- */
        size_t off = start + flen / 2 - nfft / 2; /* center window */
        double frame_band_sum = 0.0;
        double frame_band[nbands];
        for (unsigned b = 0; b < nbands; b++) {
            size_t k0 = (size_t)(blow[b]);
            size_t k1 = (size_t)(blow[b + 1]);
            if (k0 < 1) k0 = 1;
            if (k1 > nbin) k1 = nbin;
            double acc = 0.0;
            for (size_t k = k0; k < k1; k++) {
                double re = 0.0, im = 0.0;
                for (size_t i = 0; i < nfft; i++) {
                    long idx = (long)(off + i); /* symmetric clamp */
                    if (idx < 0) idx = 0;
                    if (idx >= (long)n) idx = (long)n - 1;
                    double sv = (double)samples[idx] * window[i];
                    size_t a = (i * k) % nfft;
                    re += sv * costab[a];
                    im -= sv * sintab[a];
                }
                double mag = sqrt(re * re + im * im);
                acc += mag * mag;
            }
            frame_band[b] = acc;
            frame_band_sum += acc;
        }
        if (frame_band_sum > 1e-9) {
            voiced_frames += 1.0;
            for (unsigned b = 0; b < nbands; b++)
                band[b] += frame_band[b] / frame_band_sum;
        }
    }
    free(sintab); free(costab); free(window);
    free(blow); free(bcenter);

    if (!any_signal) {
        free(rms); free(zcr); free(band);
        return SP_ERR_SILENCE;
    }
    if (voiced_frames < 1.0) {
        free(rms); free(zcr); free(band);
        return SP_ERR_SILENCE;
    }

    /* Embedding layout:
     *   [0,2) mean/std of frame RMS  (loudness dynamics)
     *   [2,4) mean/std of zero-crossing rate (voicing)
     *   [4,dim) mean log-frequency energy profile, converted to dB and
     *           centered per clip. This average spectral envelope is the
     *           primary speaker cue and is insensitive to global gain.
     */
    fbuf outb = { NULL, 0, 0 };
    sp_status_t st;
    const float *feats[2] = { rms, zcr };
    for (int j = 0; j < 2; j++) {
        double mean = 0.0, var = 0.0;
        for (size_t i = 0; i < nfr; i++) mean += feats[j][i];
        mean /= nfr;
        for (size_t i = 0; i < nfr; i++) {
            double d = feats[j][i] - mean;
            var += d * d;
        }
        if ((st = fbuf_push(&outb, (float)mean)) != SP_OK ||
            (st = fbuf_push(&outb, (float)sqrt(var / nfr))) != SP_OK) {
            free(rms); free(zcr); free(band);
            free(outb.data);
            return st;
        }
    }
    free(rms); free(zcr);

    double *prof = (double *)malloc(nbands * sizeof(double));
    if (!prof) { free(outb.data); return SP_ERR_NOMEM; }
    double pmean = 0.0;
    for (unsigned b = 0; b < nbands; b++) {
        prof[b] = 10.0 * log10(1e-10 + band[b] / voiced_frames);
        pmean += prof[b];
    }
    pmean /= nbands;
    for (unsigned b = 0; b < nbands; b++) {
        float v = (float)((prof[b] - pmean) * 0.5);
        st = fbuf_push(&outb, v);
        if (st != SP_OK) {
            free(band);
            free(prof);
            free(outb.data);
            return st;
        }
    }
    free(band);
    free(prof);

    /* Global L2 normalisation so cosine similarity is a plain dot
     * product and per-block scales are comparable. */
    double gnorm = 0.0;
    for (size_t i = 0; i < outb.len; i++) gnorm += outb.data[i] * outb.data[i];
    gnorm = sqrt(gnorm);
    if (gnorm > 1e-12f)
        for (size_t i = 0; i < outb.len; i++) outb.data[i] = (float)(outb.data[i] / gnorm);

    sp_embedding_t *e = (sp_embedding_t *)calloc(1, sizeof(*e));
    if (!e) { free(outb.data); return SP_ERR_NOMEM; }
    e->data = outb.data;
    e->dim  = outb.len;
    *out = e;
    return SP_OK;

}

sp_status_t sp_embed(const sp_model_t *m,
                     const float *samples, size_t n_samples,
                     sp_embedding_t **out)
{
    if (model_check(m) != SP_OK || !out) return SP_ERR_ARGUMENT;
    if (!samples && n_samples > 0)       return SP_ERR_ARGUMENT;

    /* Copy host PCM: the host buffer must never be read after this call,
     * and ownership of the working buffer is unambiguously ours. */
    float *copy = NULL;
    if (n_samples > 0) {
        copy = (float *)malloc(n_samples * sizeof(float));
        if (!copy) return SP_ERR_NOMEM;
        memcpy(copy, samples, n_samples * sizeof(float));
    }
    sp_status_t st = compute_embedding(m, copy, n_samples, out);
    free(copy);
    return st;
}

sp_status_t sp_embed_from_stream(const sp_model_t *m,
                                 sp_read_fn read_fn, void *read_user,
                                 sp_log_fn log_fn, void *log_user,
                                 sp_embedding_t **out)
{
    if (model_check(m) != SP_OK || !read_fn || !out)
        return SP_ERR_ARGUMENT;

    fbuf audio = { NULL, 0, 0 };
    float chunk[1024];
    for (;;) {
        int r = read_fn(read_user, chunk, 1024);
        if (r < 0) {
            free(audio.data);
            return SP_ERR_IO;
        }
        if (r == 0) break;
        for (int i = 0; i < r; i++) {
            sp_status_t st = fbuf_push(&audio, chunk[i]);
            if (st != SP_OK) { free(audio.data); return st; }
        }
    }

    if (log_fn) {
        char msg[128];
        snprintf(msg, sizeof(msg), "stream complete: %zu samples", audio.len);
        log_fn(1, msg, log_user);
    }

    /* Callback pointers and user_data are used only above; nothing is
     * stored past this point. */
    sp_status_t st = compute_embedding(m, audio.data, audio.len, out);
    free(audio.data);
    return st;
}

sp_status_t sp_similarity(const sp_model_t *m,
                          const sp_embedding_t *a, const sp_embedding_t *b,
                          float *out_score)
{
    if (model_check(m) != SP_OK || !a || !b || !out_score)
        return SP_ERR_ARGUMENT;
    if (a->dim != b->dim || a->dim != m->dim || !a->data || !b->data)
        return SP_ERR_DIM_MISMATCH;

    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a->dim; i++) {
        dot += (double)a->data[i] * b->data[i];
        na  += (double)a->data[i] * a->data[i];
        nb  += (double)b->data[i] * b->data[i];
    }
    if (na < 1e-12 || nb < 1e-12)
        return SP_ERR_SILENCE;
    double score = dot / (sqrt(na) * sqrt(nb));
    if (score < -1.0) score = -1.0;
    if (score >  1.0) score =  1.0;
    *out_score = (float)score;
    return SP_OK;
}
