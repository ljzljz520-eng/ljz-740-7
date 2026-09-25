/*
 * voiceprint.h - C ABI for the speaker-embedding (voiceprint) library.
 *
 * Ownership rules (the Go binding relies on them):
 *   - Every sp_model_t* / sp_embedding_t* returned by this library is
 *     heap allocated. The caller MUST release it with sp_model_close()
 *     or sp_embedding_free().
 *   - Input PCM buffers are copied (or fully consumed during the call).
 *     The library never retains a pointer supplied by the host beyond the
 *     return of the call.
 *   - String pointers returned by sp_strerror() are static/thread-local;
 *     callers must not free or retain them across calls.
 *   - Stream callbacks are invoked only for the duration of
 *     sp_embed_from_stream(); no callback pointer is stored afterwards.
 */
#ifndef VOICEPRINT_H
#define VOICEPRINT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SP_MAGIC   0x56505254u /* "VPRT" */
#define SP_VERSION 1u

/* Status codes. sp_strerror() converts them to human-readable text. */
typedef enum sp_status {
    SP_OK              = 0,
    SP_ERR_ARGUMENT    = 1,
    SP_ERR_NOMEM       = 2,
    SP_ERR_IO          = 3,
    SP_ERR_BAD_MODEL   = 4,
    SP_ERR_TOO_SHORT   = 5,
    SP_ERR_SILENCE     = 6,
    SP_ERR_DIM_MISMATCH = 7
} sp_status_t;

typedef struct sp_model     sp_model_t;
typedef struct sp_embedding sp_embedding_t;

struct sp_embedding {
    float  *data;
    size_t  dim;
};

/* ---- model lifecycle ------------------------------------------------ */
sp_status_t sp_model_open_file(const char *path, sp_model_t **out);
sp_status_t sp_model_open_default(sp_model_t **out);
void        sp_model_close(sp_model_t *m);

uint32_t    sp_model_sample_rate(const sp_model_t *m);
uint32_t    sp_model_dim(const sp_model_t *m);
sp_status_t sp_write_default_model(const char *path);

/* ---- feature extraction --------------------------------------------- */
sp_status_t sp_embed(const sp_model_t *m,
                     const float *samples, size_t n_samples,
                     sp_embedding_t **out);

/* streaming: the reader callback supplies PCM floats on demand.
 * user_data may be NULL and is never dereferenced by the library.
 * log_fn, if non-NULL, is used for progress diagnostics. */
typedef int (*sp_read_fn)(void *user_data, float *buf, size_t n);
typedef void (*sp_log_fn)(int level, const char *msg, void *user_data);

sp_status_t sp_embed_from_stream(const sp_model_t *m,
                                 sp_read_fn read_fn, void *read_user,
                                 sp_log_fn log_fn, void *log_user,
                                 sp_embedding_t **out);

/* ---- comparison / release ------------------------------------------- */
sp_status_t sp_similarity(const sp_model_t *m,
                          const sp_embedding_t *a, const sp_embedding_t *b,
                          float *out_score);
void        sp_embedding_free(sp_embedding_t *e);

const char *sp_strerror(sp_status_t code);

#ifdef __cplusplus
}
#endif
#endif
