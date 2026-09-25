package voiceprint

/*
#cgo CFLAGS: -I${SRCDIR}/c
#include <stdlib.h>
#include "voiceprint.h"
*/
import "C"

import (
	"io"
	"runtime/cgo"
	"unsafe"
)

type cgoHandle = cgo.Handle

func cgoNewHandle(v any) cgo.Handle { return cgo.NewHandle(v) }
func cgoDeleteHandle(h cgo.Handle)  { h.Delete() }

// goVoiceprintRead implements sp_read_fn. C owns buf; we fill it with
// at most n values and never keep a reference to buf.
//
//export goVoiceprintRead
func goVoiceprintRead(userData unsafe.Pointer, buf *C.float, n C.size_t) C.int {
	h := cgo.Handle(uintptr(userData))
	sc := h.Value().(*streamCtx)
	if sc.err != nil {
		return -1
	}

	want := int(n)
	// View the C scratch buffer as a Go slice for this callback frame
	// only; the slice header never escapes and C keeps the storage.
	tmp := unsafe.Slice((*float32)(unsafe.Pointer(buf)), want)

	got := 0
	for got < want {
		k, err := sc.r.ReadFloat32(tmp[got:])
		if k > 0 {
			got += k
		}
		if err != nil {
			if err == io.EOF {
				break
			}
			sc.err = err
			if got == 0 {
				return -1
			}
			break
		}
		if k == 0 {
			// Contract-violating source (0, nil): do not let the
			// native side spin; surface an error on the next call.
			sc.err = io.ErrNoProgress
			break
		}
	}
	return C.int(got)
}

// goVoiceprintLog implements sp_log_fn. The message pointer is owned by
// C and copied by C.GoString; it is not retained.
//
//export goVoiceprintLog
func goVoiceprintLog(level C.int, msg *C.char, userData unsafe.Pointer) {
	h := cgo.Handle(uintptr(userData))
	fn := h.Value().(func(string))
	fn(C.GoString(msg))
}
