// Package voiceprint 是声纹特征库（libvoiceprint）的 cgo 绑定。
//
// 它封装了模型加载、声纹特征提取、相似度比较、资源释放以及错误码到
// Go error 的转换。
//
// # 内存所有权约定
//
// C 库在特征提取时返回的缓冲区，Go 侧复制成普通的 []float32 后立即用
// vp_free 释放；传入 C 的音频/特征缓冲均为“调用期间临时分配、返回即释放”。
// 底层不会长期持有任何 Go 分配的内存。
//
// 日志回调通过 cgo.Handle（一个 uintptr 数值，非 Go 指针）关联，C 侧保存
// 的只是代码段中的静态跳板函数指针和该数值；回调收到的字符串在回调返回
// 前被复制为 Go string，之后不再引用。
//
// 注意：Logger 回调在 C 调用栈上执行，回调内不要再次调用同一个 Engine 的
// 方法（Engine 方法是串行化的，重入会自锁），也不要长期保存其参数。
package voiceprint

/*
#cgo LDFLAGS: -lm
#include <stdlib.h>
#include "voiceprint.h"
#include <stdint.h>

// Go 导出函数（定义在 Go 侧），先在此声明供桥接文件使用。
// 注意：//export 生成的原型不带 const，这里必须保持一致。
extern void goVPLogCallback(int level, char *msg, uintptr_t user_data);

// 桥接包装（定义在 cgo_bridge.c）：Go 侧只传一个整数 handle。
extern void vp_bridge_set_logger(vp_engine *e, uintptr_t handle);
*/
import "C"

import (
	"errors"
	"fmt"
	"runtime"
	"runtime/cgo"
	"sync"
	"unsafe"
)

// Dim 是声纹特征向量的维度。
const Dim = C.VP_DIM

// DefaultSampleRate 是内置模型期望的输入采样率（Hz）。
const DefaultSampleRate = 16000

// 底层错误码常量，与 voiceprint.h 保持一致。
const (
	codeMemory = C.VP_ERR_NOMEM
	codePath   = C.VP_ERR_PATH
	codeFormat = C.VP_ERR_FORMAT
	codeShort  = C.VP_ERR_SHORT
	codeDim    = C.VP_ERR_DIM
	codeHandle = C.VP_ERR_HANDLE
)

// ErrClosed 在引擎已释放后继续使用时返回。
var ErrClosed = errors.New("voiceprint: engine already closed")

// Logger 是底层日志回调。level 含义由底层库定义（0 为普通信息）。
// 该函数在 cgo 调用栈上被同步调用，msg 只在回调期间有效，
// 需要长期使用请自行拷贝（Go 的字符串赋值即拷贝）。
type Logger func(level int, msg string)

// Embedding 是一段音频的声纹特征向量。
type Embedding []float32

// Error 是底层库返回的错误，包含数字错误码与文字描述。
type Error struct {
	Code int
	Msg  string
}

func (e *Error) Error() string {
	return fmt.Sprintf("voiceprint: %s (code %d)", e.Msg, e.Code)
}

// Engine 是一个已加载的声纹模型句柄。
// 多个 Engine 彼此独立；单个 Engine 的方法通过互斥锁串行化。
type Engine struct {
	mu        sync.Mutex
	handle    *C.vp_engine
	logHandle cgo.Handle // 0 表示未设置日志回调
	closed    bool
	closeOnce sync.Once
}

// Open 加载模型并返回引擎。modelPath 为空时加载内置默认模型。
// 返回的 Engine 必须调用 Close 释放；GC 仅作兜底，不应依赖。
func Open(modelPath string) (*Engine, error) {
	var cPath *C.char
	if modelPath != "" {
		cPath = C.CString(modelPath)
		// 路径字符串仅在本次调用内需要，返回后立即释放。
		defer C.free(unsafe.Pointer(cPath))
	}

	var handle *C.vp_engine
	var errBuf [256]C.char

	rc := C.vp_create(cPath, &handle, &errBuf[0], C.size_t(len(errBuf)))
	if rc != C.VP_OK {
		return nil, newCError(rc, &errBuf[0])
	}

	e := &Engine{handle: handle}
	runtime.SetFinalizer(e, (*Engine).finalize)
	return e, nil
}

// Close 释放引擎及底层全部资源。可重复调用，第二次起为空操作。
func (e *Engine) Close() error {
	e.mu.Lock()
	if e.closed {
		e.mu.Unlock()
		return nil
	}
	e.closed = true
	handle := e.handle
	logHandle := e.logHandle
	e.mu.Unlock()

	runtime.SetFinalizer(e, nil)
	e.closeOnce.Do(func() {
		// vp_destroy 返回后底层不再持有回调与任何缓冲区，
		// 此刻才能删除 handle。
		C.vp_destroy(handle)
		if logHandle != 0 {
			logHandle.Delete()
		}
	})
	return nil
}

// finalize 是“忘记 Close”时的 GC 兜底，正常路径不依赖它。
func (e *Engine) finalize() {
	e.closeOnce.Do(func() {
		C.vp_destroy(e.handle)
		if e.logHandle != 0 {
			e.logHandle.Delete()
		}
	})
}

// SampleRate 返回引擎期望的输入采样率。
func (e *Engine) SampleRate() (int, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return 0, ErrClosed
	}
	return int(C.vp_sample_rate(e.handle)), nil
}

// Extract 从单声道 float32 PCM（取值建议 [-1,1]，采样率与模型一致）
// 提取声纹特征。返回的 Embedding 为 Go 侧独立内存。
func (e *Engine) Extract(pcm []float32) (Embedding, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil, ErrClosed
	}
	if len(pcm) == 0 {
		return nil, &Error{Code: int(codeFormat), Msg: "empty pcm buffer"}
	}

	n := C.size_t(len(pcm))

	// 临时 C 缓冲：复制一份给底层，函数返回前释放，底层不留引用。
	cBuf, freeBuf := floatsToC(pcm)
	defer freeBuf()

	var cEmb *C.float
	var errBuf [256]C.char

	rc := C.vp_embed(e.handle, (*C.float)(cBuf), n,
		&cEmb, &errBuf[0], C.size_t(len(errBuf)))
	// 确保 pcm 在整个 cgo 调用结束前都存活。
	runtime.KeepAlive(pcm)
	if rc != C.VP_OK {
		return nil, newCError(rc, &errBuf[0])
	}
	// 无论后续路径如何，先登记释放底层返回的特征缓冲。
	defer C.vp_free(unsafe.Pointer(cEmb))

	// 立即把底层内存复制到 Go 堆，随后 defer 释放 C 缓冲。
	emb := make(Embedding, Dim)
	src := unsafe.Slice((*float32)(unsafe.Pointer(cEmb)), Dim)
	copy(emb, src)
	return emb, nil
}

// Compare 比较两个特征向量，返回余弦相似度，范围 [-1,1]：
// 越接近 1 越可能是同一说话人。
func (e *Engine) Compare(a, b Embedding) (float32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return 0, ErrClosed
	}
	if len(a) != Dim || len(b) != Dim {
		return 0, &Error{Code: int(codeDim),
			Msg: fmt.Sprintf("embedding dimension mismatch: got %d and %d, want %d",
				len(a), len(b), Dim)}
	}

	cA, freeA := floatsToC(a)
	defer freeA()
	cB, freeB := floatsToC(b)
	defer freeB()

	var score C.float
	var errBuf [256]C.char

	rc := C.vp_compare(e.handle,
		(*C.float)(cA), C.size_t(len(a)),
		(*C.float)(cB), C.size_t(len(b)),
		&score, &errBuf[0], C.size_t(len(errBuf)))
	runtime.KeepAlive(a)
	runtime.KeepAlive(b)
	if rc != C.VP_OK {
		return 0, newCError(rc, &errBuf[0])
	}
	return float32(score), nil
}

// CompareAudio 是“提取 + 比较”的便捷封装。
func (e *Engine) CompareAudio(pcmA, pcmB []float32) (float32, error) {
	fa, err := e.Extract(pcmA)
	if err != nil {
		return 0, fmt.Errorf("extract first audio: %w", err)
	}
	fb, err := e.Extract(pcmB)
	if err != nil {
		return 0, fmt.Errorf("extract second audio: %w", err)
	}
	return e.Compare(fa, fb)
}

// SetLogger 设置底层日志回调；传 nil 清除回调。
// 替换或清除回调时，旧的关联句柄会被立即删除，避免泄漏。
func (e *Engine) SetLogger(lf Logger) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return ErrClosed
	}

	// 先在 C 侧摘除旧回调，再删除旧句柄，确保 C 不会拿到失效句柄。
	if e.logHandle != 0 {
		C.vp_bridge_set_logger(e.handle, 0)
		e.logHandle.Delete()
		e.logHandle = 0
	}

	if lf != nil {
		h := cgo.NewHandle(lf)
		// 只把整数 handle 交给 C；C 保存的是静态跳板指针 + 数值，
		// 不含任何 Go 指针（符合 cgo 指针传递规则）。
		C.vp_bridge_set_logger(e.handle, C.uintptr_t(h))
		e.logHandle = h
	}
	return nil
}

// Version 返回底层库版本字符串。
func Version() string {
	return C.GoString(C.vp_version())
}

//export goVPLogCallback
func goVPLogCallback(level C.int, msg *C.char, userData C.uintptr_t) {
	h := cgo.Handle(uintptr(userData))
	lf, ok := h.Value().(Logger)
	if !ok {
		return
	}
	// C.GoString 会把数据复制到 Go 堆；回调返回后 C 栈消息失效也无影响。
	lf(int(level), C.GoString(msg))
}

// ---- 内部工具 ----

// floatsToC 分配一块 C 内存并把 Go float32 切片复制进去。
// 返回的释放函数必须且只需调用一次。
func floatsToC(s []float32) (unsafe.Pointer, func()) {
	n := len(s)
	size := C.size_t(n) * C.size_t(unsafe.Sizeof(C.float(0)))
	p := C.malloc(size)
	if n > 0 {
		dst := unsafe.Slice((*float32)(p), n)
		copy(dst, s)
	}
	return p, func() { C.free(p) }
}

// newCError 把 C 错误码和错误缓冲拼成 Go Error；
// 缓冲为空文字时回退到错误码的默认描述。
func newCError(rc C.int, buf *C.char) error {
	msg := C.GoString(buf)
	if msg == "" {
		msg = defaultMessage(int(rc))
	}
	return &Error{Code: int(rc), Msg: msg}
}

func defaultMessage(code int) string {
	switch code {
	case int(codeMemory):
		return "out of memory"
	case int(codePath):
		return "model path error"
	case int(codeFormat):
		return "invalid data format"
	case int(codeShort):
		return "audio too short"
	case int(codeDim):
		return "dimension mismatch"
	case int(codeHandle):
		return "invalid engine handle"
	default:
		return "unknown error"
	}
}
