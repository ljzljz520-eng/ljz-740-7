// Package voiceprint is a cgo binding for the speaker-embedding
// ("voiceprint") C library (see c/voiceprint.h).
//
// The binding covers the full C ABI lifecycle:
//
//   - LoadModel / DefaultModel / WriteDefaultModel: load (or generate)
//     an acoustic model.
//   - Model.Embed / EmbedReader: turn PCM float samples into an
//     Embedding, either from a slice or from a FloatReader stream
//     (streaming uses callbacks).
//   - Model.Similarity: cosine similarity between two embeddings.
//   - Model.Close / Embedding.Close: release native resources.
//
// # Memory-safety contract
//
// C never holds Go memory across a call:
//
//   - PCM slices are copied into a C-allocated buffer for the duration
//     of the call and freed before the Go method returns.
//   - The streaming reader callback copies each chunk into a C scratch
//     buffer; Go readers and loggers are referenced through cgo.Handle
//     values (integers, not Go pointers) that are deleted as soon as the
//     native call returns. No callback or user_data is retained.
//   - Native objects (models, embeddings) live in C memory and are
//     released explicitly via Close (a runtime finalizer is only a
//     safety net).
package voiceprint

/*
#cgo CFLAGS: -I${SRCDIR}/c -Wall
#cgo linux LDFLAGS: -lm
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "voiceprint.h"

// Prototypes of the Go callbacks implemented in callbacks_cgo.go.
// Declared here manually because cgo parses each file's preamble
// independently; signatures must match the //export definitions.
extern int  goVoiceprintRead(void *user_data, float *buf, size_t n);
extern void goVoiceprintLog(int level, const char *msg, void *user_data);

// Shim so Go passes cgo.Handle values as plain uintptr_t integers; the
// integer-to-pointer cast happens in C, never as unsafe.Pointer(uintptr).
static sp_status_t sp_embed_stream_shim(
    const sp_model_t *m,
    sp_read_fn read_fn, uintptr_t read_user,
    sp_log_fn log_fn, uintptr_t log_user,
    sp_embedding_t **out) {
    return sp_embed_from_stream(m, read_fn, (void *)read_user,
                                log_fn, (void *)log_user, out);
}
*/
import "C"

import (
	"errors"
	"runtime"
	"unsafe"
)

// Code mirrors the native sp_status_t enumeration.
type Code C.int

const (
	OK             Code = C.SP_OK
	ErrArgument    Code = C.SP_ERR_ARGUMENT
	ErrNoMem       Code = C.SP_ERR_NOMEM
	ErrIO          Code = C.SP_ERR_IO
	ErrBadModel    Code = C.SP_ERR_BAD_MODEL
	ErrTooShort    Code = C.SP_ERR_TOO_SHORT
	ErrSilence     Code = C.SP_ERR_SILENCE
	ErrDimMismatch Code = C.SP_ERR_DIM_MISMATCH
)

// Error is the native error type. Its text comes from sp_strerror(),
// so messages live entirely in the C library (single source of truth).
type Error struct {
	code Code
}

func (e *Error) Error() string {
	return C.GoString(C.sp_strerror(C.sp_status_t(e.code)))
}

// Code returns the native status code.
func (e *Error) Code() Code { return e.code }

// As lets errors.As(&err, target *Error) succeed.
func (e *Error) As(target any) bool {
	if t, ok := target.(**Error); ok {
		*t = e
		return true
	}
	return false
}

// status converts a native status to nil or *Error.
func status(s C.sp_status_t) error {
	if s == C.SP_OK {
		return nil
	}
	return &Error{code: Code(s)}
}

// Model is a handle to a native acoustic model.
type Model struct {
	ptr *C.sp_model_t
}

// Embedding is a handle to a native speaker feature vector.
type Embedding struct {
	ptr *C.sp_embedding_t
	vec []float32 // copy of the native vector; native storage is freed by Close
}

// LoadModel loads a model file.
func LoadModel(path string) (*Model, error) {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath)) // do not retain the path

	var p *C.sp_model_t
	if err := status(C.sp_model_open_file(cpath, &p)); err != nil {
		return nil, err
	}
	m := &Model{ptr: p}
	runtime.SetFinalizer(m, func(x *Model) { x.Close() })
	return m, nil
}

// DefaultModel returns the built-in model (16 kHz, 48-dim embedding).
func DefaultModel() (*Model, error) {
	var p *C.sp_model_t
	if err := status(C.sp_model_open_default(&p)); err != nil {
		return nil, err
	}
	m := &Model{ptr: p}
	runtime.SetFinalizer(m, func(x *Model) { x.Close() })
	return m, nil
}

// WriteDefaultModel writes the built-in model to path so it can later
// be passed to LoadModel.
func WriteDefaultModel(path string) error {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	return status(C.sp_write_default_model(cpath))
}

// SampleRate reports the model's expected PCM sample rate in Hz.
func (m *Model) SampleRate() int {
	if m.ptr == nil {
		return 0
	}
	return int(C.sp_model_sample_rate(m.ptr))
}

// Dim reports the embedding feature dimensionality.
func (m *Model) Dim() int {
	if m.ptr == nil {
		return 0
	}
	return int(C.sp_model_dim(m.ptr))
}

// Close releases the native model. It is safe to call more than once.
func (m *Model) Close() error {
	if m.ptr == nil {
		return nil
	}
	C.sp_model_close(m.ptr)
	m.ptr = nil
	runtime.SetFinalizer(m, nil)
	return nil
}

// Embed extracts a voiceprint from PCM float samples in [-1, 1].
// The slice is copied into native memory and is never retained beyond
// this call.
func (m *Model) Embed(samples []float32) (*Embedding, error) {
	if m.ptr == nil {
		return nil, errors.New("voiceprint: model closed")
	}
	if len(samples) == 0 {
		return nil, &Error{code: ErrArgument}
	}

	n := C.size_t(len(samples))
	buf := C.malloc(n * C.size_t(unsafe.Sizeof(float32(0))))
	if buf == nil {
		return nil, &Error{code: ErrNoMem}
	}
	// Copy: C keeps a pointer to its own allocation, never to Go memory.
	C.memcpy(buf, unsafe.Pointer(&samples[0]),
		n*C.size_t(unsafe.Sizeof(float32(0))))
	defer C.free(buf)

	var out *C.sp_embedding_t
	if err := status(C.sp_embed(m.ptr,
		(*C.float)(buf), n, &out)); err != nil {
		return nil, err
	}
	return takeEmbedding(out)
}

// FloatReader is the streaming PCM source consumed by EmbedReader.
// ReadFloat32 follows the io.Reader convention: it returns 0, io.EOF
// when no samples remain, and may return n>0 with an error.
type FloatReader interface {
	ReadFloat32(p []float32) (n int, err error)
}

// EmbedReader extracts a voiceprint by streaming PCM from r through a
// native read callback. The FloatReader and logger are only alive for
// the duration of the call.
func (m *Model) EmbedReader(r FloatReader, logf func(string)) (*Embedding, error) {
	if m.ptr == nil {
		return nil, errors.New("voiceprint: model closed")
	}
	if r == nil {
		return nil, &Error{code: ErrArgument}
	}

	// cgo.Handle values are integers handed to C as user_data; C never
	// receives a Go pointer and cannot pin Go memory.
	sc := &streamCtx{r: r}
	readHandle := C.uintptr_t(cgoNewHandle(sc))
	defer cgoDeleteHandle(cgoHandle(readHandle))

	var logHandle C.uintptr_t
	var logCB C.sp_log_fn
	if logf != nil {
		logHandle = C.uintptr_t(cgoNewHandle(logf))
		defer cgoDeleteHandle(cgoHandle(logHandle))
		logCB = C.sp_log_fn(C.goVoiceprintLog)
	}

	var out *C.sp_embedding_t
	if err := status(C.sp_embed_stream_shim(
		m.ptr,
		C.sp_read_fn(C.goVoiceprintRead), readHandle,
		logCB, logHandle,
		&out)); err != nil {
		return nil, err
	}
	if sc.err != nil {
		return nil, sc.err
	}
	return takeEmbedding(out)
}

// takeEmbedding copies the native vector into Go memory and frees the
// native embedding immediately. After it returns no C buffer remains.
func takeEmbedding(p *C.sp_embedding_t) (*Embedding, error) {
	dim := int(p.dim)
	vec := unsafe.Slice((*float32)(unsafe.Pointer(p.data)), dim)
	goVec := make([]float32, dim)
	copy(goVec, vec)
	C.sp_embedding_free(p)
	return &Embedding{ptr: nil, vec: goVec}, nil
}

// Vector returns a copy of the embedding features.
func (e *Embedding) Vector() []float32 {
	if e == nil {
		return nil
	}
	out := make([]float32, len(e.vec))
	copy(out, e.vec)
	return out
}

// Dim returns the feature dimensionality.
func (e *Embedding) Dim() int { return len(e.vec) }

// Close is retained for API symmetry; embeddings returned by this
// binding have already been copied out of native memory.
func (e *Embedding) Close() error {
	if e != nil {
		e.ptr = nil
		e.vec = nil
	}
	return nil
}

// Similarity returns the cosine similarity in [-1, 1] between two
// embeddings produced by this model. The vectors are copied into native
// memory for the call and freed afterwards.
func (m *Model) Similarity(a, b *Embedding) (float32, error) {
	if m.ptr == nil {
		return 0, errors.New("voiceprint: model closed")
	}
	if a == nil || b == nil || len(a.vec) != len(b.vec) {
		return 0, &Error{code: ErrDimMismatch}
	}
	n := C.size_t(len(a.vec))
	sz := n * C.size_t(unsafe.Sizeof(float32(0)))

	ba := C.malloc(sz)
	bb := C.malloc(sz)
	if ba == nil || bb == nil {
		C.free(ba)
		C.free(bb)
		return 0, &Error{code: ErrNoMem}
	}
	defer C.free(ba)
	defer C.free(bb)
	C.memcpy(ba, unsafe.Pointer(&a.vec[0]), sz)
	C.memcpy(bb, unsafe.Pointer(&b.vec[0]), sz)

	ca := C.struct_sp_embedding{data: (*C.float)(ba), dim: n}
	cb := C.struct_sp_embedding{data: (*C.float)(bb), dim: n}

	var score C.float
	if err := status(C.sp_similarity(m.ptr, &ca, &cb, &score)); err != nil {
		return 0, err
	}
	return float32(score), nil
}

// streamCtx carries the FloatReader and the first read error through
// the callback handle.
type streamCtx struct {
	r   FloatReader
	err error
}
