/*
 * voiceprint.c — 参考实现（可替换为真实声纹引擎，ABI 保持不变）
 *
 * 特征：把整段音频分成 32 个等长子带，每个子带取 RMS 与平均过零率，
 * 再乘以模型增益并做 L2 归一化。比较使用余弦相似度。
 */
#include "voiceprint.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BANDS 32
#define MIN_SAMPLES 256

struct vp_engine {
    float gains[VP_DIM];
    int sample_rate;
    vp_log_fn log_fn;
    void *log_user;
};

static void set_err(char *buf, size_t len, const char *msg) {
    if (buf && len > 0) {
        snprintf(buf, len, "%s", msg);
    }
}

static void emit_log(vp_engine *e, int level, const char *fmt, ...) {
    if (!e->log_fn) return;
    char stack_msg[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(stack_msg, sizeof(stack_msg), fmt, ap);
    va_end(ap);
    /* msg 在栈上：仅回调期间有效，回调返回即失效，库不保存 */
    e->log_fn(level, stack_msg, e->log_user);
}

int vp_create(const char *model_path, vp_engine **out,
              char *errbuf, size_t errbuf_len) {
    if (!out) return VP_ERR_HANDLE;
    vp_engine *e = (vp_engine *)calloc(1, sizeof(vp_engine));
    if (!e) {
        set_err(errbuf, errbuf_len, "out of memory: cannot allocate engine");
        return VP_ERR_NOMEM;
    }
    e->sample_rate = 16000;

    for (int i = 0; i < VP_DIM; i++) e->gains[i] = 1.0f;

    if (model_path && *model_path) {
        FILE *f = fopen(model_path, "rb");
        if (!f) {
            set_err(errbuf, errbuf_len, "model not found or unreadable");
            free(e);
            return VP_ERR_PATH;
        }
        unsigned int magic = 0;
        if (fread(&magic, sizeof(magic), 1, f) != 1 || magic != 0x56504d44u) {
            set_err(errbuf, errbuf_len, "invalid model file (bad magic)");
            fclose(f);
            free(e);
            return VP_ERR_FORMAT;
        }
        int dim = 0;
        if (fread(&dim, sizeof(dim), 1, f) != 1 || dim != VP_DIM ||
            fread(e->gains, sizeof(float), (size_t)VP_DIM, f) != (size_t)VP_DIM) {
            set_err(errbuf, errbuf_len, "truncated or incompatible model file");
            fclose(f);
            free(e);
            return VP_ERR_FORMAT;
        }
        fclose(f);
        emit_log(e, 0, "model loaded from %s (%d dims)", model_path, VP_DIM);
    } else {
        /* 默认模型：由确定性种子生成平滑增益，仅用于演示 */
        for (int i = 0; i < VP_DIM; i++) {
            e->gains[i] = 0.75f + 0.25f * sinf((float)i * 0.37f);
        }
        emit_log(e, 0, "built-in default model loaded (%d dims)", VP_DIM);
    }

    *out = e;
    return VP_OK;
}

void vp_destroy(vp_engine *e) {
    if (!e) return;
    emit_log(e, 0, "engine destroyed");
    /* 清掉回调指针，保证销毁过程中不会再触碰外部数据 */
    e->log_fn = NULL;
    e->log_user = NULL;
    free(e);
}

int vp_sample_rate(vp_engine *e) { return e ? e->sample_rate : 0; }

void vp_free(void *p) { free(p); }

/* -------------------- 频谱特征 -------------------- */

#define FFT_N 512
#define FFT_HOP 256

static void init_hann(float *w, int n) {
	for (int i = 0; i < n; i++) {
		w[i] = 0.5f - 0.5f * cosf(2.0f * 3.14159265358979f * (float)i / (float)(n - 1));
	}
}

/* 迭代 radix-2 原地 FFT（n 必须是 2 的幂）。 */
static void fft_radix2(float *re, float *im, int n, int inverse) {
	for (int i = 1, j = 0; i < n; i++) {
		int bit = n >> 1;
		for (; j & bit; bit >>= 1) j ^= bit;
		j ^= bit;
		if (i < j) {
			float tr = re[i]; re[i] = re[j]; re[j] = tr;
			float ti = im[i]; im[i] = im[j]; im[j] = ti;
		}
	}
	for (int len = 2; len <= n; len <<= 1) {
		double ang = 2.0 * 3.14159265358979 / (double)len;
		if (inverse) ang = -ang;
		float wr = (float)cos(ang), wi = (float)sin(ang);
		for (int i = 0; i < n; i += len) {
			float cr = 1.0f, ci = 0.0f;
			for (int k = 0; k < len / 2; k++) {
				int u = i + k, v = i + k + len / 2;
				float tr = cr * re[v] - ci * im[v];
				float ti = cr * im[v] + ci * re[v];
				re[v] = re[u] - tr; im[v] = im[u] - ti;
				re[u] += tr;      im[u] += ti;
				float ncr = cr * wr - ci * wi;
				ci = cr * wi + ci * wr;
				cr = ncr;
			}
		}
	}
	if (inverse) {
		for (int i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
	}
}

int vp_embed(vp_engine *e, const float *samples, size_t n,
             float **emb_out, char *errbuf, size_t errbuf_len) {
	if (!e || !emb_out) return VP_ERR_HANDLE;
	if (!samples || n == 0) {
		set_err(errbuf, errbuf_len, "empty audio buffer");
		return VP_ERR_FORMAT;
	}
	if (n < MIN_SAMPLES) {
		if (errbuf && errbuf_len > 0) {
			snprintf(errbuf, errbuf_len,
			         "audio too short: %zu samples, need >= %d",
			         n, MIN_SAMPLES);
		}
		return VP_ERR_SHORT;
	}

	float *emb = (float *)malloc(sizeof(float) * VP_DIM);
	float *win = (float *)malloc(sizeof(float) * FFT_N);
	float *re = (float *)malloc(sizeof(float) * FFT_N);
	float *im = (float *)calloc((size_t)FFT_N, sizeof(float));
	if (!emb || !win || !re || !im) {
		free(emb); free(win); free(re); free(im);
		set_err(errbuf, errbuf_len, "out of memory: cannot allocate work buffers");
		return VP_ERR_NOMEM;
	}
	init_hann(win, FFT_N);

	/* 对所有帧累加：前 32 维=子带对数功率；后 32 维=子带能量重心（谱形状）。 */
	double band_pow[BANDS];
	double band_cent[BANDS];
	for (int b = 0; b < BANDS; b++) { band_pow[b] = 0.0; band_cent[b] = 0.0; }

	long frames = 0;
	for (size_t start = 0; start + FFT_N <= n; start += FFT_HOP) {
		for (int i = 0; i < FFT_N; i++) {
			re[i] = samples[start + (size_t)i] * win[i];
			im[i] = 0.0f;
		}
		fft_radix2(re, im, FFT_N, 0);

		int half = FFT_N / 2 + 1;
		double total_pow[BANDS], total_cf[BANDS];
		for (int b = 0; b < BANDS; b++) { total_pow[b] = 0.0; total_cf[b] = 0.0; }

		for (int k = 0; k < half; k++) {
			double p = (double)re[k] * re[k] + (double)im[k] * im[k];
			int b = k * BANDS / half;
			if (b >= BANDS) b = BANDS - 1;
			total_pow[b] += p;
			total_cf[b] += p * (double)k;   /* 用 bin 索引加权的重心 */
		}
		for (int b = 0; b < BANDS; b++) {
			band_pow[b] += total_pow[b];
			if (total_pow[b] > 1e-9) band_cent[b] += total_cf[b] / total_pow[b];
		}
		frames++;
	}

	if (frames <= 0) {
		free(emb); free(win); free(re); free(im);
		set_err(errbuf, errbuf_len, "audio shorter than one analysis window");
		return VP_ERR_SHORT;
	}

	/* 对对数功率谱减去跨子带均值，只保留谱包络形状（消除响度/直流基线）。 */
	double lp[BANDS], mean = 0.0;
	for (int b = 0; b < BANDS; b++) {
		lp[b] = log10(1.0 + 1e6 * band_pow[b] / (double)frames);
		mean += lp[b];
	}
	mean /= BANDS;

	double norm = 0.0;
	for (int b = 0; b < BANDS; b++) {
		double cf = band_cent[b] / (double)frames / (double)(FFT_N / 2);
		emb[b] = (float)((lp[b] - mean) * 2.0);
		emb[BANDS + b] = (float)(cf * 2.0);
		emb[b] *= e->gains[b];
		emb[BANDS + b] *= e->gains[BANDS + b];
		norm += (double)emb[b] * emb[b] +
		        (double)emb[BANDS + b] * emb[BANDS + b];
	}
	free(win); free(re); free(im);

	if (norm < 1e-12) {
		free(emb);
		set_err(errbuf, errbuf_len, "silence input: feature norm is zero");
		return VP_ERR_FORMAT;
	}
	float scale = 1.0f / (float)sqrt(norm);
	for (int i = 0; i < VP_DIM; i++) emb[i] *= scale;

	*emb_out = emb;
	return VP_OK;
}

int vp_compare(vp_engine *e,
               const float *a, size_t na,
               const float *b, size_t nb,
               float *score, char *errbuf, size_t errbuf_len) {
    (void)e;
    if (!score) return VP_ERR_HANDLE;
    if (na != (size_t)VP_DIM || nb != (size_t)VP_DIM || !a || !b) {
        set_err(errbuf, errbuf_len, "embedding dimension mismatch");
        return VP_ERR_DIM;
    }
    double dot = 0.0, na2 = 0.0, nb2 = 0.0;
    for (size_t i = 0; i < VP_DIM; i++) {
        dot += (double)a[i] * b[i];
        na2 += (double)a[i] * a[i];
        nb2 += (double)b[i] * b[i];
    }
    double denom = sqrt(na2) * sqrt(nb2);
    if (denom < 1e-12) {
        set_err(errbuf, errbuf_len, "zero embedding norm");
        return VP_ERR_FORMAT;
    }
    double s = dot / denom;
    if (s > 1.0) s = 1.0;
    if (s < -1.0) s = -1.0;
    *score = (float)s;
    return VP_OK;
}

void vp_set_logger(vp_engine *e, vp_log_fn fn, void *user_data) {
    if (!e) return;
    e->log_fn = fn;
    e->log_user = user_data;
}

const char *vp_version(void) { return "1.0.0"; }
