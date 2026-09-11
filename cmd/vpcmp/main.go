// vpcmp compares two audio files and prints their voiceprint similarity.
//
// Usage:
//
//	vpcmp [-model model.bin] <a.wav> <b.wav>
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"example.com/voiceprint"
)

func main() {
	model := flag.String("model", "model.bin", "voiceprint model file")
	flag.Parse()
	args := flag.Args()
	if len(args) != 2 {
		fmt.Fprintf(os.Stderr, "usage: vpcmp [-model model.bin] <a.wav> <b.wav>\n")
		os.Exit(2)
	}

	eng, err := voiceprint.LoadModel(*model)
	if err != nil {
		fatal(err)
	}
	defer eng.Close()

	vec := make([][]float32, 2)
	for i, p := range args {
		samples, rate, err := readWAV(p)
		if err != nil {
			fatal(fmt.Errorf("%s: %w", p, err))
		}
		name := filepath.Base(p)
		v, err := eng.ExtractWithProgress(samples, rate, func(fr float64) {
			fmt.Fprintf(os.Stderr, "\rextracting %-20s %6.2f%%", name, fr*100)
			if fr >= 1 {
				fmt.Fprintln(os.Stderr)
			}
		})
		if err != nil {
			fatal(fmt.Errorf("%s: %w", p, err))
		}
		vec[i] = v
	}

	sim, err := voiceprint.Similarity(vec[0], vec[1])
	if err != nil {
		fatal(err)
	}
	fmt.Printf("similarity: %.4f\n", sim)
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "vpcmp:", err)
	os.Exit(1)
}
