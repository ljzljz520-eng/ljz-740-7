package voiceprint

import (
	"math"
	"os"
	"path/filepath"
	"sync"
	"testing"
)

func sine(freq float64, sampleRate, n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(0.8 * math.Sin(2*math.Pi*freq*float64(i)/float64(sampleRate)))
	}
	return out
}

func newTestEngine(t *testing.T) *Engine {
	t.Helper()
	model := filepath.Join(t.TempDir(), "model.bin")
	if err := os.WriteFile(model, []byte("fake voiceprint model weights"), 0o644); err != nil {
		t.Fatal(err)
	}
	eng, err := LoadModel(model)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	t.Cleanup(func() { eng.Close() })
	return eng
}

func TestLoadModelError(t *testing.T) {
	_, err := LoadModel(filepath.Join(t.TempDir(), "nope.bin"))
	if err == nil {
		t.Fatal("expected error for missing model")
	}
	if err.Error() == "" {
		t.Fatal("error text must not be empty")
	}
}

func TestExtractAndSimilarity(t *testing.T) {
	eng := newTestEngine(t)
	if eng.Dim() <= 0 {
		t.Fatalf("bad dim %d", eng.Dim())
	}
	const sr = 16000
	va, err := eng.Extract(sine(440, sr, sr), sr)
	if err != nil {
		t.Fatalf("Extract: %v", err)
	}
	if len(va) != eng.Dim() {
		t.Fatalf("got %d dims, want %d", len(va), eng.Dim())
	}
	vb, err := eng.Extract(sine(440, sr, sr), sr)
	if err != nil {
		t.Fatal(err)
	}
	vc, err := eng.Extract(sine(1730, sr, sr), sr)
	if err != nil {
		t.Fatal(err)
	}

	same, err := Similarity(va, vb)
	if err != nil {
		t.Fatal(err)
	}
	if same < 0.999 {
		t.Fatalf("identical signals: similarity %f, want ~1", same)
	}
	diff, err := Similarity(va, vc)
	if err != nil {
		t.Fatal(err)
	}
	if diff >= same {
		t.Fatalf("different tones: similarity %f should be < %f", diff, same)
	}
	t.Logf("same=%.4f different=%.4f", same, diff)
}

func TestProgressCallback(t *testing.T) {
	eng := newTestEngine(t)
	var fracs []float64
	_, err := eng.ExtractWithProgress(sine(440, 16000, 16000), 16000, func(f float64) {
		fracs = append(fracs, f)
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(fracs) == 0 {
		t.Fatal("progress callback never fired")
	}
	for _, f := range fracs {
		if f <= 0 || f > 1 {
			t.Fatalf("progress %f out of (0,1]", f)
		}
	}
	if fracs[len(fracs)-1] != 1 {
		t.Fatalf("last progress = %f, want 1", fracs[len(fracs)-1])
	}
}

func TestClose(t *testing.T) {
	eng := newTestEngine(t)
	if err := eng.Close(); err != nil {
		t.Fatal(err)
	}
	if err := eng.Close(); err != nil {
		t.Fatal(err) // idempotent
	}
	if _, err := eng.Extract(sine(440, 16000, 1600), 16000); err == nil {
		t.Fatal("Extract after Close must fail")
	}
}

func TestSimilarityErrors(t *testing.T) {
	if _, err := Similarity([]float32{1, 2}, []float32{1}); err == nil {
		t.Fatal("length mismatch must fail")
	}
	if _, err := Similarity([]float32{0, 0}, []float32{0, 0}); err == nil {
		t.Fatal("zero-norm must fail")
	}
}

func TestConcurrentExtract(t *testing.T) {
	eng := newTestEngine(t)
	var wg sync.WaitGroup
	for g := 0; g < 4; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := eng.Extract(sine(440, 16000, 4000), 16000); err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
}
