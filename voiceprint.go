// Package voiceprint provides Go bindings for the native "vp" voiceprint
// feature library: load a model, extract embeddings, compare similarity,
// release resources, and read error text.
//
// Memory ownership policy (so the native side never holds Go memory):
//
//   - Every []float32 handed to C is first copied into C.malloc'ed memory
//     that is freed as soon as the call returns; the native library never
//     receives a pointer into the Go heap.
//   - Strings (model path, error text) are copied across the boundary with
//     C.CString / C.GoString; no buffer is shared in either direction.
//   - The progress callback crosses the boundary as a runtime/cgo.Handle
//     (an integer, not a Go pointer) that is deleted immediately after the
//     native call returns, so neither the Go func value nor its closure
//     can be retained by the library.
//   - The native engine handle is released by Close, with a finalizer as
//     a backstop only.
//
// If a future native API is documented to never retain its inputs, the
// input copies may be replaced by runtime.Pinner to avoid them.
package voiceprint

/*
#cgo LDFLAGS: -lm

#include <stdint.h>
#include <stdlib.h>
#include "vp.h"

// Defined in bridge.c: the trampoline that calls back into Go, and a
// helper that packs a cgo.Handle into the void* user slot (so the Go
// side never converts an integer into unsafe.Pointer itself).
vp_progress_fn vp_progress_callback(void);
void *vp_user_from_handle(uintptr_t h);
*/
import "C"

import (
	"errors"
	"fmt"
	"math"
	"runtime"
	"runtime/cgo"
	"sync"
	"unsafe"
)

// errBufLen is the size of the C-owned buffer used to receive error text.
const errBufLen = 256

// Engine is a loaded voiceprint model. It is safe for concurrent use;
// native calls are serialized internally.
type Engine struct {
	mu  sync.Mutex
	ptr *C.vp_engine_t
	dim int
}

// LoadModel loads the model file at path.
// Call Close when the engine is no longer needed.
func LoadModel(path string) (*Engine, error) {
	cpath := C.CString(path) // copied into C memory
	defer C.free(unsafe.Pointer(cpath))

	errp := C.malloc(errBufLen)
	if errp == nil {
		return nil, errors.New("voiceprint: out of memory")
	}
	defer C.free(errp)
	*(*C.char)(errp) = 0

	ptr := C.vp_engine_create(cpath, (*C.char)(errp), errBufLen)
	if ptr == nil {
		return nil, goErr(errp, "voiceprint: failed to load model")
	}
	e := &Engine{ptr: ptr, dim: int(C.vp_engine_dim(ptr))}
	// Backstop only; deterministic cleanup goes through Close.
	runtime.SetFinalizer(e, func(e *Engine) { e.Close() })
	return e, nil
}

// Dim returns the embedding dimension of the loaded model.
func (e *Engine) Dim() int { return e.dim }

// Close releases the native engine. It is idempotent and safe to call
// concurrently with other methods (they will fail once it completes).
func (e *Engine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.ptr != nil {
		C.vp_engine_destroy(e.ptr)
		e.ptr = nil
	}
	runtime.SetFinalizer(e, nil)
	return nil
}

// LastError returns the library's last error message for the calling
// thread. The C string is copied before returning; the native buffer is
// never referenced afterwards. Prefer the error values returned by this
// package's functions, which are captured at the exact point of failure.
func LastError() string {
	msg := C.vp_last_error()
	if msg == nil {
		return ""
	}
	return C.GoString(msg)
}

// Extract computes the voiceprint embedding of mono PCM samples
// (float32 in [-1, 1], sampleRate in 8000..48000).
func (e *Engine) Extract(samples []float32, sampleRate int) ([]float32, error) {
	return e.extract(samples, sampleRate, nil)
}

// ExtractWithProgress is Extract plus a progress callback (fraction 0..1).
// The callback runs on the goroutine that called ExtractWithProgress and
// is never invoked after it returns.
func (e *Engine) ExtractWithProgress(samples []float32, sampleRate int, onProgress func(frac float64)) ([]float32, error) {
	return e.extract(samples, sampleRate, onProgress)
}

func (e *Engine) extract(samples []float32, sampleRate int, onProgress func(float64)) ([]float32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.ptr == nil {
		return nil, errors.New("voiceprint: engine is closed")
	}
	if len(samples) == 0 {
		return nil, errors.New("voiceprint: no samples")
	}
	if len(samples) > math.MaxInt32 {
		return nil, errors.New("voiceprint: too many samples")
	}

	n := len(samples)
	floatSize := C.size_t(unsafe.Sizeof(C.float(0)))

	// Copy the PCM into C-owned memory: the native library only ever sees
	// memory it owns, never the Go heap, and it is freed on return.
	cin := C.malloc(C.size_t(n) * floatSize)
	if cin == nil {
		return nil, errors.New("voiceprint: out of memory")
	}
	defer C.free(cin)
	copy(unsafe.Slice((*float32)(cin), n), samples)

	dim := e.dim
	cout := C.malloc(C.size_t(dim) * floatSize)
	if cout == nil {
		return nil, errors.New("voiceprint: out of memory")
	}
	defer C.free(cout)

	errp := C.malloc(errBufLen)
	if errp == nil {
		return nil, errors.New("voiceprint: out of memory")
	}
	defer C.free(errp)
	*(*C.char)(errp) = 0

	var cb C.vp_progress_fn
	var user unsafe.Pointer
	if onProgress != nil {
		// Only the handle (an integer) crosses the boundary. It is deleted
		// as soon as the native call returns, so C cannot retain anything
		// that points at Go memory.
		h := cgo.NewHandle(onProgress)
		defer h.Delete()
		cb = C.vp_progress_callback()
		user = C.vp_user_from_handle(C.uintptr_t(h))
	}

	rc := C.vp_engine_extract(e.ptr, (*C.float)(cin), C.int(n), C.int(sampleRate),
		(*C.float)(cout), C.int(dim), cb, user, (*C.char)(errp), errBufLen)
	if rc != 0 {
		return nil, goErr(errp, "voiceprint: extract failed")
	}

	// Copy the embedding out; the C buffer is freed by the defer above.
	out := make([]float32, dim)
	copy(out, unsafe.Slice((*float32)(cout), dim))
	return out, nil
}

//export vpGoProgress
func vpGoProgress(frac C.double, user unsafe.Pointer) {
	if user == nil {
		return
	}
	fn, ok := cgo.Handle(user).Value().(func(float64))
	if !ok || fn == nil {
		return
	}
	fn(float64(frac))
}

// Similarity returns the cosine similarity in [-1, 1] of two embeddings
// produced by the same model.
func Similarity(a, b []float32) (float32, error) {
	if len(a) == 0 || len(a) != len(b) {
		return 0, fmt.Errorf("voiceprint: embedding length mismatch (%d vs %d)", len(a), len(b))
	}
	if len(a) > math.MaxInt32 {
		return 0, errors.New("voiceprint: embeddings too large")
	}
	var na, nb float64
	for i := range a {
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0, errors.New("voiceprint: zero-norm embedding")
	}

	n := len(a)
	size := C.size_t(n) * C.size_t(unsafe.Sizeof(C.float(0)))

	ca := C.malloc(size)
	if ca == nil {
		return 0, errors.New("voiceprint: out of memory")
	}
	defer C.free(ca)
	cb := C.malloc(size)
	if cb == nil {
		return 0, errors.New("voiceprint: out of memory")
	}
	defer C.free(cb)
	copy(unsafe.Slice((*float32)(ca), n), a)
	copy(unsafe.Slice((*float32)(cb), n), b)

	return float32(C.vp_cosine((*C.float)(ca), (*C.float)(cb), C.int(n))), nil
}

// goErr builds a Go error from a C error buffer, copying the text out.
func goErr(errp unsafe.Pointer, fallback string) error {
	s := C.GoString((*C.char)(errp))
	if s == "" {
		s = fallback
	}
	return errors.New(s)
}
