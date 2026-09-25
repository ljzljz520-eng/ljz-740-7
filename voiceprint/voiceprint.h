/*
 * voiceprint.h — 声纹特征提取 C 接口（底层库 ABI）
 *
 * 线程安全说明：同一个 vp_engine 句柄的调用需外部串行化；
 * 不同句柄之间互相独立。
 */
#ifndef VOICEPRINT_H
#define VOICEPRINT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VP_OK           0
#define VP_ERR_NOMEM   -1
#define VP_ERR_PATH    -2
#define VP_ERR_FORMAT  -3
#define VP_ERR_SHORT   -4   /* 音频过短 */
#define VP_ERR_DIM     -5   /* 特征维度不匹配 */
#define VP_ERR_HANDLE  -6

#define VP_DIM 64   /* 特征维度：32 个频带 RMS + 32 个过零率 */

typedef struct vp_engine vp_engine;

/* 日志回调：msg 内存由库持有，仅在回调期间有效，回调返回后不得使用 */
typedef void (*vp_log_fn)(int level, const char *msg, void *user_data);

/*
 * 加载模型。model_path 为 NULL 或空串时使用内置默认模型。
 * 成功返回 0，失败返回错误码并把文字写入 errbuf（容量 errbuf_len）。
 */
int vp_create(const char *model_path, vp_engine **out,
              char *errbuf, size_t errbuf_len);

/* 释放引擎及其全部资源。e==NULL 时为空操作。 */
void vp_destroy(vp_engine *e);

/* 引擎期望的输入采样率（调用方需先重采样）。 */
int vp_sample_rate(vp_engine *e);

/*
 * 提取声纹特征。
 * samples   : float32 PCM，取值 [-1,1]，长度 n
 * emb_out   : 库内部分配、长度 VP_DIM 的 float32 数组；
 *             调用方必须用 vp_free 释放，避免长期占用。
 */
int vp_embed(vp_engine *e, const float *samples, size_t n,
             float **emb_out, char *errbuf, size_t errbuf_len);

/* 释放 vp_embed 返回的缓冲区。 */
void vp_free(void *p);

/* 余弦相似度，返回 [-1,1]。维度不一致返回 VP_ERR_DIM。 */
int vp_compare(vp_engine *e,
               const float *a, size_t na,
               const float *b, size_t nb,
               float *score, char *errbuf, size_t errbuf_len);

/* 设置/清除日志回调（user_data 原样透传）。 */
void vp_set_logger(vp_engine *e, vp_log_fn fn, void *user_data);

const char *vp_version(void);

#ifdef __cplusplus
}
#endif

#endif /* VOICEPRINT_H */
