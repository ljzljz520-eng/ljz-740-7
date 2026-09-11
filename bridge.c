/*
 * Callback trampoline: cgo cannot take the address of a Go-exported
 * function from Go code, so this tiny C shim hands the Go side a plain
 * function pointer that forwards into the //export'ed Go function.
 */
#include "vp.h"

#include <stdint.h>

/* Implemented in Go (see //export vpGoProgress in voiceprint.go). */
extern void vpGoProgress(double frac, void *user);

static void vp_progress_trampoline(double frac, void *user) {
    vpGoProgress(frac, user);
}

/* Returns the function pointer to register as vp_progress_fn. */
vp_progress_fn vp_progress_callback(void) { return vp_progress_trampoline; }

/*
 * Packs a Go cgo.Handle (arrived as a plain integer) into the void*
 * user slot. Doing the conversion here keeps the Go side free of any
 * uintptr -> unsafe.Pointer conversion.
 */
void *vp_user_from_handle(uintptr_t h) { return (void *)h; }
