# voiceprint — Go bindings for a speaker-embedding ("voiceprint") C library

`example.com/voiceprint` wraps a small C ABI (`c/voiceprint.h`, implementation
`voiceprint_c.c`) with a memory-safe cgo layer:

| C concern            | Go API                                           |
| -------------------- | ------------------------------------------------ |
| load model           | `LoadModel(path)` / `DefaultModel()`             |
| extract features     | `Model.Embed([]float32)` / `Model.EmbedReader()` |
| compare similarity   | `Model.Similarity(a, b)` → `[-1, 1]`             |
| release resources    | `Model.Close()` / `Embedding.Close()`            |
| native error text    | `*voiceprint.Error`, text from `sp_strerror()`   |

The bundled C implementation is a self-contained DSP reference (energy /
zero-crossing moments + a mean log-frequency spectral envelope compared with
cosine similarity). Replace `voiceprint_c.c` with a production speaker
engine — keep `c/voiceprint.h` unchanged and no Go code needs to move.

## Quick start

Requires cgo (a C compiler). Input audio: mono WAV (8/16/24/32-bit PCM or
32-bit float) at the model sample rate (built-in model: 16 kHz).

```sh
# compare two voices
go run ./cmd/voiceprint a.wav b.wav
# similarity: 0.9993
# verdict: same speaker (>= 0.50)

# streaming extractor (callbacks) with native logs
go run ./cmd/voiceprint -stream -v a.wav b.wav

# persist / load the built-in model
go run ./cmd/voiceprint -init -model model.bin
go run ./cmd/voiceprint -model model.bin a.wav b.wav
```

Exit code is `2` when the score is below `-threshold`, `1` on errors.

## Library use

```go
m, err := voiceprint.DefaultModel()          // or LoadModel("model.bin")
if err != nil { log.Fatal(err) }
defer m.Close()

a, err := m.Embed(pcmA)                       // []float32, range [-1, 1]
b, err := m.Embed(pcmB)
score, err := m.Similarity(a, b)             // -1..1, higher = same speaker

// streaming via callback:
emb, err := m.EmbedReader(reader, func(msg string) { log.Print(msg) })
```

`DecodeWAV(io.Reader)` is provided for decoding WAV files to
`(samples []float32, sampleRate int)`.

## Memory safety: C never pins Go memory

cgo forbids storing Go pointers long-term in C; this binding is built around
copy-in / copy-out and integer handles:

- **Input PCM** (`Embed`, `Similarity`) is copied with `C.malloc`/`C.memcpy`
  into C-owned buffers that are `C.free`d before the Go call returns. The
  library's `sp_embed()` itself also copies, so no host buffer is ever read
  after a call.
- **Embeddings** returned from C are immediately copied into a Go slice and
  the native `sp_embedding_t` is freed (`takeEmbedding`); Go code never holds
  a C buffer.
- **Streaming callbacks** (`EmbedReader`): the reader/logger are passed to C
  as `cgo.Handle` values (integers), never as Go pointers. Handles are
  `Delete()`d via `defer` as soon as the native call finishes, and C stores
  neither callback nor `user_data` afterwards. The read callback writes into
  the C-owned scratch buffer through an `unsafe.Slice` view that never
  escapes the callback frame. A tiny C shim
  (`sp_embed_stream_shim(uintptr_t, ...)`) performs the integer→pointer cast
  on the C side so `go vet` stays happy.
- **String arguments** use `CString` + `defer C.free`; `sp_strerror` returns
  static strings copied with `C.GoString`, never freed or retained by Go.
- Models are native handles released by `Close` (idempotent, guarded); a
  runtime finalizer is a safety net only — `TestResourceReclaimUnderGC`
  exercises the finalizer path under `-race`.

## Layout

```
c/voiceprint.h      C ABI (status codes, lifecycle, callbacks)
voiceprint_c.c      reference DSP implementation
voiceprint.go       cgo binding: model/embedding lifecycle, errors
callbacks_cgo.go    //export Go read/log callbacks + cgo.Handle bridge
wav.go              pure-Go WAV decoder
cmd/voiceprint/     example CLI: two WAVs -> similarity score
```
