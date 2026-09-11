package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
)

// readWAV decodes a RIFF/WAVE file (PCM 8/16-bit or float32) into mono
// float32 samples in [-1, 1] and returns them with the sample rate.
func readWAV(path string) ([]float32, int, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, 0, err
	}
	if len(b) < 12 || string(b[0:4]) != "RIFF" || string(b[8:12]) != "WAVE" {
		return nil, 0, fmt.Errorf("not a RIFF/WAVE file")
	}
	var (
		format, channels, bits int
		sampleRate             int
		data                   []byte
	)
	for off := 12; off+8 <= len(b); {
		id := string(b[off : off+4])
		size := int(binary.LittleEndian.Uint32(b[off+4 : off+8]))
		body := off + 8
		if body+size > len(b) {
			size = len(b) - body
		}
		switch id {
		case "fmt ":
			if size < 16 {
				return nil, 0, fmt.Errorf("malformed fmt chunk")
			}
			format = int(binary.LittleEndian.Uint16(b[body:]))
			channels = int(binary.LittleEndian.Uint16(b[body+2:]))
			sampleRate = int(binary.LittleEndian.Uint32(b[body+4:]))
			bits = int(binary.LittleEndian.Uint16(b[body+14:]))
		case "data":
			data = b[body : body+size]
		}
		off = body + size + (size & 1) // chunks are word-aligned
	}
	if data == nil || channels <= 0 || sampleRate <= 0 {
		return nil, 0, fmt.Errorf("missing fmt/data chunk")
	}

	var frames []float32
	switch {
	case format == 1 && bits == 16:
		n := len(data) / 2
		frames = make([]float32, n)
		for i := 0; i < n; i++ {
			frames[i] = float32(int16(binary.LittleEndian.Uint16(data[2*i:]))) / 32768
		}
	case format == 1 && bits == 8:
		frames = make([]float32, len(data))
		for i, v := range data {
			frames[i] = (float32(v) - 128) / 128
		}
	case format == 3 && bits == 32:
		n := len(data) / 4
		frames = make([]float32, n)
		for i := 0; i < n; i++ {
			frames[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[4*i:]))
		}
	default:
		return nil, 0, fmt.Errorf("unsupported WAV format (format=%d bits=%d)", format, bits)
	}

	if channels > 1 { // downmix to mono
		n := len(frames) / channels
		mono := make([]float32, n)
		for i := 0; i < n; i++ {
			var s float32
			for c := 0; c < channels; c++ {
				s += frames[i*channels+c]
			}
			mono[i] = s / float32(channels)
		}
		frames = mono
	}
	return frames, sampleRate, nil
}
