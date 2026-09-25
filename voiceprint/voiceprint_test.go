package voiceprint

import (
	"errors"
	"math"
	"os"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
)

func makePCM(freq float64, n int, sr float64) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(0.5 * math.Sin(2*math.Pi*freq*float64(i)/sr))
	}
	return out
}

func TestVersion(t *testing.T) {
	if Version() == "" {
		t.Fatal("empty version")
	}
}

func TestOpenBadModel(t *testing.T) {
	if _, err := Open("/definitely/not/here.vpm"); err == nil {
		t.Fatal("expected error for missing model")
	} else {
		var ce *Error
		if !errors.As(err, &ce) || ce.Code != int(codePath) {
			t.Fatalf("want VP_ERR_PATH *Error, got %T %v", err, err)
		}
		if !strings.Contains(err.Error(), "code -2") {
			t.Fatalf("error text missing code: %v", err)
		}
	}
}

func TestExtractAndCompare(t *testing.T) {
	e, err := Open("")
	if err != nil {
		t.Fatalf("Open default: %v", err)
	}
	defer e.Close()

	sr := 16000
	if got, err := e.SampleRate(); err != nil || got != sr {
		t.Fatalf("SampleRate = %d, %v; want %d", got, err, sr)
	}

	a := makePCM(120, sr*2, float64(sr))
	b := makePCM(123, sr*2, float64(sr))
	c := makePCM(260, sr*2, float64(sr))

	fa, err := e.Extract(a)
	if err != nil {
		t.Fatalf("Extract a: %v", err)
	}
	if len(fa) != Dim {
		t.Fatalf("dim = %d, want %d", len(fa), Dim)
	}

	// 同一信号自比必须为 1（归一化后的余弦）。
	self, err := e.Compare(fa, fa)
	if err != nil {
		t.Fatalf("Compare self: %v", err)
	}
	if math.Abs(float64(self)-1) > 1e-5 {
		t.Fatalf("self score = %v, want 1", self)
	}

	fb, _ := e.Extract(b)
	fc, _ := e.Extract(c)

	simSame, err := e.Compare(fa, fb)
	if err != nil {
		t.Fatalf("Compare close pitch: %v", err)
	}
	simDiff, err := e.Compare(fa, fc)
	if err != nil {
		t.Fatalf("Compare far pitch: %v", err)
	}
	if !(simSame > simDiff) {
		t.Fatalf("expected same-speaker %.4f > diff %.4f", simSame, simDiff)
	}

	// 便捷封装结果一致。
	direct, err := e.CompareAudio(a, b)
	if err != nil || math.Abs(float64(direct-simSame)) > 1e-6 {
		t.Fatalf("CompareAudio = %v, %v; want ~%v", direct, err, simSame)
	}
}

func TestExtractInvalidInput(t *testing.T) {
	e, _ := Open("")
	defer e.Close()

	if _, err := e.Extract(nil); err == nil {
		t.Fatal("expected error on empty pcm")
	}
	if _, err := e.Extract(make([]float32, 16)); err == nil {
		t.Fatal("expected error on too-short pcm")
	}

	fa, _ := e.Extract(makePCM(100, 4000, 16000))
	if _, err := e.Compare(fa, make([]float32, 10)); err == nil {
		t.Fatal("expected dimension error")
	}
}

func TestCloseIdempotentAndUseAfterClose(t *testing.T) {
	e, _ := Open("")
	if err := e.Close(); err != nil {
		t.Fatalf("first close: %v", err)
	}
	if err := e.Close(); err != nil {
		t.Fatalf("second close: %v", err)
	}
	if _, err := e.Extract(makePCM(100, 4000, 16000)); !errors.Is(err, ErrClosed) {
		t.Fatalf("Extract after close err = %v, want ErrClosed", err)
	}
	if err := e.SetLogger(nil); !errors.Is(err, ErrClosed) {
		t.Fatalf("SetLogger after close err = %v", err)
	}
}

func TestLoggerCallback(t *testing.T) {
	e, _ := Open("")

	var mu sync.Mutex
	var msgs []string
	if err := e.SetLogger(func(level int, msg string) {
		mu.Lock()
		defer mu.Unlock()
		msgs = append(msgs, msg)
	}); err != nil {
		t.Fatalf("SetLogger: %v", err)
	}

	// 触发一次提取，底层当前不在该路径打日志；改为替换回调确认通路。
	got := make(chan string, 1)
	if err := e.SetLogger(func(level int, msg string) {
		select {
		case got <- msg:
		default:
		}
	}); err != nil {
		t.Fatalf("replace logger: %v", err)
	}

	if _, err := e.Extract(makePCM(120, 4000, 16000)); err != nil {
		t.Fatalf("Extract: %v", err)
	}
	// Close 时底层会发一条 "engine destroyed"。
	if err := e.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	select {
	case m := <-got:
		if !strings.Contains(m, "destroyed") {
			t.Fatalf("unexpected log: %q", m)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("logger callback was not invoked")
	}
}

func TestEmbeddingIndependentAfterReuse(t *testing.T) {
	// 连续提取多次，确认后一次调用不影响前一次拿到的 Go 副本，
	// 且得分稳定（间接验证 C 缓冲已及时释放、无悬挂引用）。
	e, _ := Open("")
	defer e.Close()

	a := makePCM(120, 4000, 16000)
	b := makePCM(300, 4000, 16000)

	fa, err := e.Extract(a)
	if err != nil {
		t.Fatal(err)
	}
	snap := append(Embedding(nil), fa...)
	for i := 0; i < 50; i++ {
		if _, err := e.Extract(b); err != nil {
			t.Fatal(err)
		}
	}
	for i := range snap {
		if snap[i] != fa[i] {
			t.Fatalf("embedding mutated by later calls at %d", i)
		}
	}
}

func TestFinalizerFreesForgottenEngine(t *testing.T) {
	// 不显式 Close，依赖 runtime 兜底释放（仅验证不 panic、可回收）。
	func() {
		e, err := Open("")
		if err != nil {
			t.Fatal(err)
		}
		_ = e
	}()
	for i := 0; i < 5; i++ {
		runtime.GC()
		time.Sleep(10 * time.Millisecond)
	}
}

func TestReadWAV(t *testing.T) {
	// 用一个极简 16-bit 单声道 WAV 做往返验证。
	path := t.TempDir() + "/in.wav"
	sr := 16000
	n := 4000
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	// RIFF 头（size 字段可宽松）
	header := make([]byte, 0, 44)
	header = append(header, []byte("RIFF")...)
	header = append(header, uint32Bytes(uint32(36+n*2))...)
	header = append(header, []byte("WAVEfmt ")...)
	header = append(header, 16, 0, 0, 0) // fmt size
	header = append(header, 1, 0)        // PCM
	header = append(header, 1, 0)        // mono
	header = append(header, uint32Bytes(uint32(sr))...)
	header = append(header, uint32Bytes(uint32(sr*2))...) // byte rate
	header = append(header, 2, 0)                         // block align
	header = append(header, 16, 0)                        // bits
	header = append(header, []byte("data")...)
	header = append(header, uint32Bytes(uint32(n*2))...)
	if _, err := f.Write(header); err != nil {
		t.Fatal(err)
	}
	// 写入两个相反相位的满幅采样交替
	for i := 0; i < n; i++ {
		var v int16 = 30000
		if i%2 == 1 {
			v = -30000
		}
		if _, err := f.Write(int16Bytes(v)); err != nil {
			t.Fatal(err)
		}
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	pcm, info, err := ReadWAV(path)
	if err != nil {
		t.Fatalf("ReadWAV: %v", err)
	}
	if info.SampleRate != sr || info.Channels != 1 || info.NumSamples != n {
		t.Fatalf("info = %+v", info)
	}
	if len(pcm) != n || math.Abs(float64(pcm[0])-30000.0/32768.0) > 1e-6 {
		t.Fatalf("pcm[0] = %v", pcm[0])
	}

	if _, _, err := ReadWAV("/no/such/file.wav"); err == nil {
		t.Fatal("expected open error")
	}
}

func uint32Bytes(v uint32) []byte {
	return []byte{byte(v), byte(v >> 8), byte(v >> 16), byte(v >> 24)}
}

func int16Bytes(v int16) []byte {
	u := uint16(v)
	return []byte{byte(u), byte(u >> 8)}
}

func TestConcurrentEnginesAndCalls(t *testing.T) {
	var wg sync.WaitGroup
	for g := 0; g < 4; g++ {
		wg.Add(1)
		go func(base float64) {
			defer wg.Done()
			e, err := Open("")
			if err != nil {
				t.Errorf("Open: %v", err)
				return
			}
			defer e.Close()
			pcm := makePCM(base, 8000, 16000)
			f1, err := e.Extract(pcm)
			if err != nil {
				t.Errorf("Extract: %v", err)
				return
			}
			for i := 0; i < 20; i++ {
				if _, err := e.Compare(f1, f1); err != nil {
					t.Errorf("Compare: %v", err)
					return
				}
			}
		}(100.0 + float64(g)*40)
	}
	wg.Wait()
}
