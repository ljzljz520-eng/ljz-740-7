#ifndef VP_H
#define VP_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * vp - voiceprint feature library (C ABI).
 *
 * Memory ownership contract (the Go binding relies on this):
 *  - The library never retains pointers passed by the caller; every
 *    pointer argument is only dereferenced for the duration of the call.
 *  - Error text is reported through caller-provided buffers (err/err_cap).
 *    vp_last_error() additionally mirrors the last message into
 *    thread-local storage owned by the library; callers must copy it.
 *  - The progress callback is invoked synchronously from
 *    vp_engine_extract(); neither `cb` nor `user` is kept after the call.
 */

typedef struct vp_engine vp_engine_t;

/* Optional extraction progress callback; frac grows from 0 to 1. */
typedef void (*vp_progress_fn)(double frac, void *user);

/*
 * Loads the model file at model_path.
 * Returns NULL on failure; err (if non-NULL) receives the message.
 */
vp_engine_t *vp_engine_create(const char *model_path, char *err, size_t err_cap);

/* Embedding dimension produced by vp_engine_extract(). */
int vp_engine_dim(const vp_engine_t *e);

/*
 * Extracts a voiceprint embedding from mono float32 PCM.
 *   samples/n:   input audio, only read during the call.
 *   sample_rate: 8000..48000.
 *   out/cap:     caller buffer, cap >= vp_engine_dim(e).
 *   cb/user:     optional progress callback, never retained.
 *   err/err_cap: optional error buffer.
 * Returns 0 on success, -1 on failure.
 */
int vp_engine_extract(vp_engine_t *e,
                      const float *samples, int n, int sample_rate,
                      float *out, int cap,
                      vp_progress_fn cb, void *user,
                      char *err, size_t err_cap);

/* Cosine similarity of two dim-length embeddings, in [-1, 1]. */
float vp_cosine(const float *a, const float *b, int dim);

/* Last error message on this thread; never NULL, library-owned. */
const char *vp_last_error(void);

/* Frees an engine; NULL is allowed. */
void vp_engine_destroy(vp_engine_t *e);

#ifdef __cplusplus
}
#endif

#endif /* VP_H */
