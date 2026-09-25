package voiceprint

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
)

// DecodeWAV reads an uncompressed PCM (8/16/24/32-bit integer) or
// 32-bit IEEE-float WAV stream, downmixes every channel to mono and
// returns samples in [-1, 1] plus the declared sample rate.
//
// It is a convenience helper: the model itself consumes raw mono float
// PCM matching Model.SampleRate().
func DecodeWAV(r io.Reader) (samples []float32, sampleRate int, err error) {
	var tag [4]byte
	if _, err = io.ReadFull(r, tag[:]); err != nil {
		return nil, 0, errors.New("voiceprint: not a RIFF file")
	}
	if string(tag[:]) != "RIFF" {
		return nil, 0, errors.New("voiceprint: not a RIFF file")
	}
	var riffSize uint32
	if err = binary.Read(r, binary.LittleEndian, &riffSize); err != nil {
		return nil, 0, err
	}
	if _, err = io.ReadFull(r, tag[:]); err != nil {
		return nil, 0, err
	}
	if string(tag[:]) != "WAVE" {
		return nil, 0, errors.New("voiceprint: not a WAVE file")
	}

	var (
		audioFormat   uint16
		channels      uint16
		rate          uint32
		bitsPerSample uint16
		fmtParsed     bool
		audio         []byte
	)

	// Walk RIFF chunks.
	for {
		var id [4]byte
		if _, err = io.ReadFull(r, id[:]); err != nil {
			if err == io.EOF || err == io.ErrUnexpectedEOF {
				break
			}
			return nil, 0, err
		}
		var size uint32
		if err = binary.Read(r, binary.LittleEndian, &size); err != nil {
			return nil, 0, err
		}
		body := make([]byte, size)
		if _, err = io.ReadFull(r, body); err != nil {
			return nil, 0, fmt.Errorf("voiceprint: truncated %q chunk: %w", id, err)
		}
		if size%2 == 1 {
			if _, err = r.Read(make([]byte, 1)); err != nil {
				return nil, 0, err
			}
		}

		switch string(id[:]) {
		case "fmt ":
			if size < 16 {
				return nil, 0, errors.New("voiceprint: fmt chunk too small")
			}
			audioFormat = binary.LittleEndian.Uint16(body[0:2])
			channels = binary.LittleEndian.Uint16(body[2:4])
			rate = binary.LittleEndian.Uint32(body[4:8])
			bitsPerSample = binary.LittleEndian.Uint16(body[14:16])
			fmtParsed = true
		case "data":
			audio = body
		}
	}

	if !fmtParsed {
		return nil, 0, errors.New("voiceprint: missing fmt chunk")
	}
	if channels == 0 {
		return nil, 0, errors.New("voiceprint: zero channels")
	}
	if len(audio) == 0 {
		return nil, 0, errors.New("voiceprint: empty data chunk")
	}

	// 1 = PCM integer, 3 = IEEE float. WAVE_FORMAT_EXTENSIBLE (0xFFFE)
	// with a float subformat is also reported as 3 by common encoders
	// for the plain tag; keep support focused on 1 and 3 here.
	switch audioFormat {
	case 1:
		switch bitsPerSample {
		case 8, 16, 24, 32:
		default:
			return nil, 0, fmt.Errorf("voiceprint: unsupported PCM bit depth %d", bitsPerSample)
		}
	case 3:
		if bitsPerSample != 32 {
			return nil, 0, fmt.Errorf("voiceprint: unsupported float bit depth %d", bitsPerSample)
		}
	default:
		return nil, 0, fmt.Errorf("voiceprint: unsupported WAV format tag %d", audioFormat)
	}

	frameBytes := int(bitsPerSample) / 8 * int(channels)
	frames := len(audio) / frameBytes
	out := make([]float32, frames)

	for f := 0; f < frames; f++ {
		var acc float64
		base := f * frameBytes
		for c := 0; c < int(channels); c++ {
			off := base + c*(int(bitsPerSample)/8)
			switch {
			case audioFormat == 1 && bitsPerSample == 8:
				// unsigned 8-bit
				acc += (float64(audio[off]) - 128.0) / 128.0
			case audioFormat == 1 && bitsPerSample == 16:
				v := int16(binary.LittleEndian.Uint16(audio[off : off+2]))
				acc += float64(v) / 32768.0
			case audioFormat == 1 && bitsPerSample == 24:
				u := uint32(audio[off]) | uint32(audio[off+1])<<8 | uint32(audio[off+2])<<16
				if u&0x800000 != 0 {
					u |= 0xff000000 // sign extend
				}
				acc += float64(int32(u)) / 8388608.0
			case audioFormat == 1 && bitsPerSample == 32:
				v := int32(binary.LittleEndian.Uint32(audio[off : off+4]))
				acc += float64(v) / 2147483648.0
			case audioFormat == 3 && bitsPerSample == 32:
				bits := binary.LittleEndian.Uint32(audio[off : off+4])
				acc += float64(math.Float32frombits(bits))
			}
		}
		s := float32(acc / float64(channels))
		if s > 1 {
			s = 1
		} else if s < -1 {
			s = -1
		}
		out[f] = s
	}
	return out, int(rate), nil
}
