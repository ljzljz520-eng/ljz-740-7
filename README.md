# voiceprint — Go bindings for the native `vp` voiceprint library

Wraps a C voiceprint feature library (cgo): load a model, extract
embeddings, compare similarity, release resources, and read error text.

## Layout

| File                | Role                                                        |
|---------------------|-------------------------------------------------------------|
| `vp.h`              | C ABI + memory-ownership contract                           |
| `vp.c`              | Deterministic stub of the native library (swap for the real one); its extraction projection is derived from the model file |
| `bridge.c`          | Callback trampoline into the `//export`ed Go function       |
| `voiceprint.go`     | The Go binding (`Engine`, `Extract`, `Similarity`, `Close`) |
| `cmd/vpcmp`         | Example: print the similarity score of two WAV files        |
| `cmd/gentone`       | Helper: generate sine-wave WAVs for the demo                |

## Go API

```go
eng, err := voiceprint.LoadModel("model.bin") // load model
defer eng.Close()                             // release native resources

v1, err := eng.Extract(samples, 16000)        // []float32 mono PCM -> embedding
v2, err := eng.ExtractWithProgress(samples, 16000, func(frac float64) {
    // called synchronously, never after Extract returns
})

sim, err := voiceprint.Similarity(v1, v2)     // cosine similarity in [-1, 1]

msg := voiceprint.LastError()                 // last native error text (copied)
```

## Memory-safety policy (no Go memory retained by the native side)

* Every `[]float32` passed to C is first copied into `C.malloc` memory and
  freed when the call returns — the library never sees Go heap pointers.
* Strings cross the boundary via `C.CString` / `C.GoString` (copies).
* Error text is read through a C-owned buffer allocated per call, so it is
  immune to goroutine thread migration (unlike a thread-local readback).
* The progress callback crosses the boundary as a `runtime/cgo.Handle`
  (an integer, not a Go pointer) that is `Delete`d right after the native
  call returns; the library cannot retain the Go func value or its closure.
* `Close` frees the native engine; a finalizer is only a backstop.

## Demo

```sh
go test ./...

head -c 2048 /dev/urandom > model.bin        # any non-empty file for the stub
go run ./cmd/gentone -freq 440  -out a.wav
go run ./cmd/gentone -freq 1400 -out b.wav
go run ./cmd/vpcmp -model model.bin a.wav a.wav   # similarity: 1.0000
go run ./cmd/vpcmp -model model.bin a.wav b.wav   # similarity: much lower
```

The stub derives its extraction projection from the model file bytes, so
the model genuinely participates in feature computation: different model
files give different embeddings for the same audio, while the same model
file is fully deterministic.
