// gentone writes a mono 16-bit PCM WAV containing a pure sine wave,
// handy for trying out vpcmp.
//
// Usage:
//
//	gentone [-freq 440] [-dur 1.0] [-sr 16000] [-amp 0.8] [-out tone.wav]
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"os"
)

func main() {
	freq := flag.Float64("freq", 440, "tone frequency in Hz")
	dur := flag.Float64("dur", 1.0, "duration in seconds")
	sr := flag.Int("sr", 16000, "sample rate in Hz")
	amp := flag.Float64("amp", 0.8, "amplitude 0..1")
	out := flag.String("out", "tone.wav", "output file")
	flag.Parse()

	f, s := *freq, *sr
	n := int(*dur * float64(s))
	data := make([]byte, 2*n)
	for i := 0; i < n; i++ {
		v := int16(*amp * 32767 * math.Sin(2*math.Pi*f*float64(i)/float64(s)))
		binary.LittleEndian.PutUint16(data[2*i:], uint16(v))
	}
	if err := os.WriteFile(*out, wavFile(1, s, 16, data), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "gentone:", err)
		os.Exit(1)
	}
	fmt.Printf("wrote %s (%.1f Hz, %d Hz, %d samples)\n", *out, f, s, n)
}

func wavFile(channels, sampleRate, bitsPerSample int, data []byte) []byte {
	byteRate := sampleRate * channels * bitsPerSample / 8
	blockAlign := channels * bitsPerSample / 8
	h := make([]byte, 44)
	copy(h[0:], "RIFF")
	binary.LittleEndian.PutUint32(h[4:], uint32(36+len(data)))
	copy(h[8:], "WAVE")
	copy(h[12:], "fmt ")
	binary.LittleEndian.PutUint32(h[16:], 16)
	binary.LittleEndian.PutUint16(h[20:], 1) // PCM
	binary.LittleEndian.PutUint16(h[22:], uint16(channels))
	binary.LittleEndian.PutUint32(h[24:], uint32(sampleRate))
	binary.LittleEndian.PutUint32(h[28:], uint32(byteRate))
	binary.LittleEndian.PutUint16(h[32:], uint16(blockAlign))
	binary.LittleEndian.PutUint16(h[34:], uint16(bitsPerSample))
	copy(h[36:], "data")
	binary.LittleEndian.PutUint32(h[40:], uint32(len(data)))
	return append(h, data...)
}
