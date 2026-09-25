package voiceprint

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
)

// WaveInfo 描述一个 WAV 文件的基本信息。
type WaveInfo struct {
	SampleRate int
	Channels   int
	Bits       int
	IsFloat    bool
	NumSamples int // 转换后的单声道采样点总数
}

// ReadWAV 读取未压缩 PCM/float WAV，混音为单声道并归一化到 [-1,1]。
// 支持：8/16/24/32 位整数 PCM 与 32 位 IEEE float。
// 不做重采样：调用方应保证采样率与引擎期望一致。
func ReadWAV(path string) ([]float32, WaveInfo, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, WaveInfo{}, fmt.Errorf("open wav: %w", err)
	}
	defer f.Close()

	var riff [12]byte
	if _, err := io.ReadFull(f, riff[:]); err != nil {
		return nil, WaveInfo{}, errors.New("not a wav file: header too short")
	}
	if string(riff[0:4]) != "RIFF" || string(riff[8:12]) != "WAVE" {
		return nil, WaveInfo{}, errors.New("not a wav file: missing RIFF/WAVE tag")
	}

	var (
		audioFormat uint16
		channels    uint16
		sampleRate  uint32
		bits        uint16
	)
	var data []byte

	for {
		var hdr [8]byte
		if _, err := io.ReadFull(f, hdr[:]); err != nil {
			if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
				break
			}
			return nil, WaveInfo{}, fmt.Errorf("read chunk header: %w", err)
		}
		id := string(hdr[0:4])
		size := binary.LittleEndian.Uint32(hdr[4:8])

		if id == "fmt " {
			chunk := make([]byte, size)
			if _, err := io.ReadFull(f, chunk); err != nil {
				return nil, WaveInfo{}, errors.New("truncated fmt chunk")
			}
			if len(chunk) < 16 {
				return nil, WaveInfo{}, errors.New("fmt chunk too small")
			}
			audioFormat = binary.LittleEndian.Uint16(chunk[0:2])
			channels = binary.LittleEndian.Uint16(chunk[2:4])
			sampleRate = binary.LittleEndian.Uint32(chunk[4:8])
			bits = binary.LittleEndian.Uint16(chunk[14:16])
		} else if id == "data" {
			data = make([]byte, size)
			if _, err := io.ReadFull(f, data); err != nil {
				return nil, WaveInfo{}, errors.New("truncated data chunk")
			}
		} else {
			// LIST/fact 等其它块：直接跳过（奇数长度补 1 字节）。
			if _, err := io.CopyN(io.Discard, f, int64(size)); err != nil {
				return nil, WaveInfo{}, fmt.Errorf("skip chunk %q: %w", id, err)
			}
		}
		if size%2 == 1 {
			if _, err := f.Seek(1, io.SeekCurrent); err != nil {
				return nil, WaveInfo{}, err
			}
		}
	}

	if data == nil {
		return nil, WaveInfo{}, errors.New("wav has no data chunk")
	}
	if channels == 0 || bits == 0 || sampleRate == 0 {
		return nil, WaveInfo{}, errors.New("wav missing valid fmt chunk")
	}
	// 1 = PCM integer；3 = IEEE float
	isFloat := audioFormat == 3
	if audioFormat != 1 && !isFloat {
		return nil, WaveInfo{}, fmt.Errorf("unsupported wav format: %d (only PCM/float)", audioFormat)
	}
	if isFloat && bits != 32 {
		return nil, WaveInfo{}, fmt.Errorf("unsupported float bit depth: %d", bits)
	}

	bytesPerSample := int(bits) / 8
	frameSize := bytesPerSample * int(channels)
	if frameSize <= 0 || len(data) < frameSize {
		return nil, WaveInfo{}, errors.New("wav data shorter than one frame")
	}
	frames := len(data) / frameSize

	out := make([]float32, frames)
	for fr := 0; fr < frames; fr++ {
		var acc float64
		base := fr * frameSize
		for ch := 0; ch < int(channels); ch++ {
			off := base + ch*bytesPerSample
			acc += float64(decodeSample(data[off:off+bytesPerSample], bits, isFloat))
		}
		out[fr] = float32(acc / float64(channels))
	}

	return out, WaveInfo{
		SampleRate: int(sampleRate),
		Channels:   int(channels),
		Bits:       int(bits),
		IsFloat:    isFloat,
		NumSamples: frames,
	}, nil
}

func decodeSample(b []byte, bits uint16, isFloat bool) float32 {
	if isFloat {
		return RawFloat32LE(b)
	}
	switch bits {
	case 8:
		// WAV 8 位 PCM 为无符号 [0,255]，中点 128。
		return (float32(b[0]) - 128.0) / 128.0
	case 16:
		v := int16(binary.LittleEndian.Uint16(b))
		return float32(v) / 32768.0
	case 24:
		u := uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16
		if u&0x800000 != 0 {
			u |= 0xff000000 // 符号扩展
		}
		return float32(int32(u)) / 8388608.0
	case 32:
		v := int32(binary.LittleEndian.Uint32(b))
		return float32(v) / 2147483648.0
	default:
		return 0
	}
}

// RawFloat32LE 把 4 个小端字节解码为 float32。
func RawFloat32LE(b []byte) float32 {
	return math.Float32frombits(binary.LittleEndian.Uint32(b))
}
