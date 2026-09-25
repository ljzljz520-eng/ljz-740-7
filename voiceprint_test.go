package voiceprint_test

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	vp "example.com/voiceprint"
)

const sr = 16000

// tone synthesises an amplitude-modulated, vibrato-shaped pseudo-speech
// signal. Different f0/modRate produce different frame statistics.
func tone(f0, modRate float64, durSec float64) []float32 {
	n := int(durSec * sr)
	s := make([]float32, n)
	for i := 0; i < n; i++ {
		t := float64(i) / sr
		env := 0.55 + 0.45*math.Sin(2*math.Pi*modRate*t)
		ph := 2*math.Pi*f0*t + 3.0*math.Sin(2*math.Pi*5*t)
		v := env * (0.7*math.Sin(ph) + 0.3*math.Sin(2*ph))
		// deterministic, signal-dependent fine noise
		v += 0.01 * math.Sin(2*math.Pi*(f0*3.7)*t)
		s[i] = float32(v)
	}
	return s
}

func TestSameVoiceScoreOne(t *testing.T) {
	m, err := vp.DefaultModel()
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	x := tone(150, 3.5, 0.6)
	a, err := m.Embed(x)
	if err != nil {
		t.Fatal(err)
	}
	b, err := m.Embed(x)
	if err != nil {
		t.Fatal(err)
	}
	score, err := m.Similarity(a, b)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(float64(score)-1) > 1e-5 {
		t.Fatalf("same signal score = %v, want 1", score)
	}
}

func TestDifferentVoicesLowerScore(t *testing.T) {
	m, _ := vp.DefaultModel()
	defer m.Close()

	a, err := m.Embed(tone(120, 3.0, 0.6))
	if err != nil {
		t.Fatal(err)
	}
	b, err := m.Embed(tone(320, 8.0, 0.6))
	if err != nil {
		t.Fatal(err)
	}
	diff, err := m.Similarity(a, b)
	if err != nil {
		t.Fatal(err)
	}
	if diff > 0.9 {
		t.Fatalf("different voices too similar: %v", diff)
	}
	if diff < -1 || diff > 1 {
		t.Fatalf("score out of range: %v", diff)
	}
	t.Logf("different-voice similarity = %.4f", diff)
}

func TestStreamMatchesSlice(t *testing.T) {
	m, _ := vp.DefaultModel()
	defer m.Close()

	x := tone(180, 4.5, 0.6)
	fromSlice, err := m.Embed(x)
	if err != nil {
		t.Fatal(err)
	}
	var logs int
	fromStream, err := m.EmbedReader(&sliceReader{s: x}, func(string) { logs++ })
	if err != nil {
		t.Fatal(err)
	}
	if logs == 0 {
		t.Fatal("logger callback never invoked")
	}
	score, err := m.Similarity(fromSlice, fromStream)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(float64(score)-1) > 1e-5 {
		t.Fatalf("stream vs slice score = %v, want 1", score)
	}
}

type sliceReader struct {
	s   []float32
	pos int
}

func (r *sliceReader) ReadFloat32(p []float32) (int, error) {
	if r.pos >= len(r.s) {
		return 0, errEOF
	}
	n := copy(p, r.s[r.pos:])
	r.pos += n
	if r.pos >= len(r.s) {
		return n, errEOF
	}
	return n, nil
}

var errEOF = io.EOF

// use a distinct sentinel to prove non-EOF reader errors propagate
type boomReader struct{ calls int }

func (boomReader) ReadFloat32(p []float32) (int, error) {
	return 0, errors.New("disk on fire")
}

func TestStreamReaderError(t *testing.T) {
	m, _ := vp.DefaultModel()
	defer m.Close()
	if _, err := m.EmbedReader(boomReader{}, nil); err == nil {
		t.Fatal("expected reader error, got nil")
	} else if err.Error() != "disk on fire" {
		// native returns IO first for -1 with no data; either branch is
		// acceptable, but it must mention a failure.
		var ne *vp.Error
		if !errors.As(err, &ne) || ne.Code() != vp.ErrIO {
			t.Fatalf("unexpected error: %v", err)
		}
	}
}

func TestNativeErrorCodesAndText(t *testing.T) {
	m, _ := vp.DefaultModel()
	defer m.Close()

	cases := []struct {
		name string
		pcm  []float32
		want vp.Code
	}{
		{"too short", make([]float32, 100), vp.ErrTooShort},
		{"silence", make([]float32, sr), vp.ErrSilence},
	}
	for _, c := range cases {
		_, err := m.Embed(c.pcm)
		var ne *vp.Error
		if !errors.As(err, &ne) {
			t.Fatalf("%s: expected *voiceprint.Error, got %T", c.name, err)
		}
		if ne.Code() != c.want {
			t.Fatalf("%s: code = %d, want %d", c.name, ne.Code(), c.want)
		}
		if ne.Error() == "" {
			t.Fatalf("%s: empty error text", c.name)
		}
		t.Logf("%s -> %q", c.name, ne.Error())
	}

	if _, err := m.Embed(nil); err == nil {
		t.Fatal("nil pcm should fail")
	}
}

func TestCloseIdempotentAndGuarded(t *testing.T) {
	m, _ := vp.DefaultModel()
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	if err := m.Close(); err != nil {
		t.Fatalf("double close: %v", err)
	}
	if _, err := m.Embed([]float32{0.1, 0.2}); err == nil {
		t.Fatal("embed after close should fail")
	}
}

func TestModelFileRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "model.bin")
	if err := vp.WriteDefaultModel(path); err != nil {
		t.Fatal(err)
	}
	m, err := vp.LoadModel(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if m.SampleRate() != 16000 || m.Dim() != 48 {
		t.Fatalf("unexpected model: rate=%d dim=%d", m.SampleRate(), m.Dim())
	}
	if _, err := vp.LoadModel(filepath.Join(dir, "missing.bin")); err == nil {
		t.Fatal("missing model should fail")
	}
	garbage := filepath.Join(dir, "bad.bin")
	if err := os.WriteFile(garbage, []byte("not a model"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := vp.LoadModel(garbage); err == nil {
		t.Fatal("garbage model should fail")
	}
}

// --- WAV decoder ------------------------------------------------------

func writeWAV16(t *testing.T, path string, rate int, channels int, mono []float32) {
	t.Helper()
	frames := len(mono)
	dataBytes := frames * channels * 2
	var buf bytes.Buffer
	put := func(v any) {
		if err := binary.Write(&buf, binary.LittleEndian, v); err != nil {
			t.Fatal(err)
		}
	}
	buf.WriteString("RIFF")
	put(uint32(36 + dataBytes))
	buf.WriteString("WAVE")
	buf.WriteString("fmt ")
	put(uint32(16))
	put(uint16(1))
	put(uint16(channels))
	put(uint32(rate))
	put(uint32(rate * channels * 2))
	put(uint16(channels * 2))
	put(uint16(16))
	buf.WriteString("data")
	put(uint32(dataBytes))
	for i := 0; i < frames; i++ {
		for c := 0; c < channels; c++ {
			// second stereo channel carries the inverted signal; mono
			// mixdown should still be ~0 for the side-only component.
			v := mono[i]
			if channels == 2 && c == 1 {
				v = -mono[i]
			}
			x := int16(math.MaxInt16 * float64(v))
			put(x)
		}
	}
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestDecodeWAVAndEndToEnd(t *testing.T) {
	dir := t.TempDir()
	x := tone(160, 4, 0.6)
	p := filepath.Join(dir, "a.wav")
	writeWAV16(t, p, sr, 1, x)

	f, err := os.Open(p)
	if err != nil {
		t.Fatal(err)
	}
	samples, rate, err := vp.DecodeWAV(f)
	f.Close()
	if err != nil {
		t.Fatal(err)
	}
	if rate != sr || len(samples) != len(x) {
		t.Fatalf("decoded rate=%d n=%d", rate, len(samples))
	}

	// stereo file should downmix to (near) zero for L=-R
	sp := filepath.Join(dir, "s.wav")
	writeWAV16(t, sp, sr, 2, x)
	sf, _ := os.Open(sp)
	mix, _, err := vp.DecodeWAV(sf)
	sf.Close()
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range mix {
		if math.Abs(float64(v)) > 1e-4 {
			t.Fatalf("stereo mid/side not centered: %v", v)
		}
	}
}

func TestResourceReclaimUnderGC(t *testing.T) {
	for i := 0; i < 200; i++ {
		m, err := vp.DefaultModel()
		if err != nil {
			t.Fatal(err)
		}
		x := tone(100+float64(i%50), 3+float64(i%4), 0.2)
		a, err := m.Embed(x)
		if err != nil {
			t.Fatal(err)
		}
		b, err := m.EmbedReader(&sliceReader{s: append([]float32(nil), x...)}, nil)
		if err != nil {
			t.Fatal(err)
		}
		if s, err := m.Similarity(a, b); err != nil || s < 0.99 {
			t.Fatalf("iter %d score=%v err=%v", i, s, err)
		}
		// Deliberately leak m/a/b to the GC on every other iteration:
		// the finalizer must reclaim native memory without crashing.
		if i%2 == 0 {
			_ = m.Close()
		}
	}
	// Trigger finalizers for all models dropped without Close.
	runtime.GC()
}
