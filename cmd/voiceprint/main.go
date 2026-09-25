// Command voiceprint loads a speaker-embedding model, extracts a
// voiceprint from each of two WAV files and prints their similarity
// score in [-1, 1] (higher = more likely the same speaker).
//
// Usage:
//
//	voiceprint [flags] a.wav b.wav
//
//	-model     path to a model file (default: built-in model)
//	-init      write the built-in model to -model path and exit
//	-stream    use the streaming (callback) extractor
//	-threshold same-speaker threshold (default 0.5)
//	-v         verbose native logs
//
// Exit code 2 means the voices are judged different (below threshold),
// which makes the command handy in scripts and CI.
package main

import (
	"flag"
	"fmt"
	"io"
	"os"

	vp "example.com/voiceprint"
)

func main() {
	modelPath := flag.String("model", "", "model file (empty = built-in)")
	initModel := flag.Bool("init", false, "write built-in model to -model path and exit")
	stream := flag.Bool("stream", false, "use streaming (callback) extractor")
	threshold := flag.Float64("threshold", 0.5, "same-speaker threshold")
	verbose := flag.Bool("v", false, "verbose native logs")
	flag.Parse()

	if *initModel {
		if *modelPath == "" {
			fatal("-init requires a -model path")
		}
		if err := vp.WriteDefaultModel(*modelPath); err != nil {
			fatalErr("write model", err)
		}
		fmt.Printf("wrote default model to %s\n", *modelPath)
		return
	}

	if flag.NArg() != 2 {
		fmt.Fprintln(os.Stderr, "usage: voiceprint [-model path] [-stream] a.wav b.wav")
		os.Exit(1)
	}

	model, err := openModel(*modelPath)
	if err != nil {
		fatalErr("load model", err)
	}
	defer model.Close()

	var logf func(string)
	if *verbose {
		logf = func(msg string) { fmt.Fprintf(os.Stderr, "[native] %s\n", msg) }
	}

	embed := func(path string) *vp.Embedding {
		samples, rate, err := readMonoWAV(path)
		if err != nil {
			fatalErr("read "+path, err)
		}
		if rate != model.SampleRate() {
			fatal(fmt.Sprintf("%s: sample rate %d does not match model %d",
				path, rate, model.SampleRate()))
		}
		var emb *vp.Embedding
		if *stream {
			emb, err = model.EmbedReader(newFloatSource(samples), logf)
		} else {
			emb, err = model.Embed(samples)
		}
		if err != nil {
			fatalErr("embed "+path, err)
		}
		return emb
	}

	a := embed(flag.Arg(0))
	defer a.Close()
	b := embed(flag.Arg(1))
	defer b.Close()

	score, err := model.Similarity(a, b)
	if err != nil {
		fatalErr("compare", err)
	}

	fmt.Printf("similarity: %.4f\n", score)
	if score < float32(*threshold) {
		fmt.Printf("verdict: different speakers (< %.2f)\n", *threshold)
		os.Exit(2)
	}
	fmt.Printf("verdict: same speaker (>= %.2f)\n", *threshold)
}

func openModel(path string) (*vp.Model, error) {
	if path != "" {
		return vp.LoadModel(path)
	}
	return vp.DefaultModel()
}

func readMonoWAV(path string) ([]float32, int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, err
	}
	defer f.Close()
	return vp.DecodeWAV(f)
}

// floatSource adapts an in-memory PCM slice to vp.FloatReader, showing
// how an application streams samples through the native callback.
type floatSource struct {
	samples []float32
	pos     int
}

func newFloatSource(s []float32) *floatSource { return &floatSource{samples: s} }

func (f *floatSource) ReadFloat32(p []float32) (int, error) {
	if f.pos >= len(f.samples) {
		return 0, io.EOF
	}
	n := copy(p, f.samples[f.pos:])
	f.pos += n
	if f.pos >= len(f.samples) {
		return n, io.EOF
	}
	return n, nil
}

func fatal(msg string) {
	fmt.Fprintln(os.Stderr, "voiceprint:", msg)
	os.Exit(1)
}

func fatalErr(stage string, err error) {
	fmt.Fprintf(os.Stderr, "voiceprint: %s: %v\n", stage, err)
	os.Exit(1)
}
