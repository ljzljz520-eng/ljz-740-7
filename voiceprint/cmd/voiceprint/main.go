// 命令 voiceprint：输入两段音频，输出声纹相似度。
//
// 用法：
//
//	voiceprint [-model path] [-v] audio1.wav audio2.wav
//
// .wav/.wave 文件按 WAV 解析（自动混音为单声道）；
// 其它扩展名按“原始 float32 小端 PCM”读取，可用 -raw-rate 指定采样率。
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"voiceprint"
)

func main() {
	modelPath := flag.String("model", "", "模型文件路径（留空用内置默认模型）")
	rawRate := flag.Int("raw-rate", voiceprint.DefaultSampleRate, "裸 PCM( float32 LE ) 的采样率")
	verbose := flag.Bool("v", false, "打开底层日志回调")
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr,
			"voiceprint (libvoiceprint %s)\n用法: %s [-model path] [-v] audio1 audio2\n",
			voiceprint.Version(), filepath.Base(os.Args[0]))
		flag.PrintDefaults()
	}
	flag.Parse()

	if flag.NArg() != 2 {
		flag.Usage()
		os.Exit(2)
	}

	engine, err := voiceprint.Open(*modelPath)
	if err != nil {
		log.Fatalf("加载模型失败: %v", err)
	}
	defer engine.Close()

	if *verbose {
		if err := engine.SetLogger(func(level int, msg string) {
			log.Printf("[vp lvl=%d] %s", level, msg)
		}); err != nil {
			log.Fatalf("设置日志回调失败: %v", err)
		}
	}

	pcm1, info1, err := loadAudio(flag.Arg(0), engine, *rawRate)
	if err != nil {
		log.Fatalf("读取第一段音频失败: %v", err)
	}
	pcm2, info2, err := loadAudio(flag.Arg(1), engine, *rawRate)
	if err != nil {
		log.Fatalf("读取第二段音频失败: %v", err)
	}

	if info1.SampleRate != info2.SampleRate {
		log.Printf("警告: 两段音频采样率不同 (%d vs %d)，结果可能不可靠",
			info1.SampleRate, info2.SampleRate)
	}

	emb1, err := engine.Extract(pcm1)
	if err != nil {
		log.Fatalf("第一段提特征失败: %v", err)
	}
	emb2, err := engine.Extract(pcm2)
	if err != nil {
		log.Fatalf("第二段提特征失败: %v", err)
	}

	score, err := engine.Compare(emb1, emb2)
	if err != nil {
		log.Fatalf("相似度比较失败: %v", err)
	}

	fmt.Printf("音频1: %s (%d Hz)\n", flag.Arg(0), info1.SampleRate)
	fmt.Printf("音频2: %s (%d Hz)\n", flag.Arg(1), info2.SampleRate)
	fmt.Printf("特征维度: %d\n", voiceprint.Dim)
	fmt.Printf("声纹相似度: %.6f\n", score)
	fmt.Printf("判定: %s\n", verdict(score))
}

func loadAudio(path string, engine *voiceprint.Engine, rawRate int) ([]float32, voiceprint.WaveInfo, error) {
	if strings.EqualFold(filepath.Ext(path), ".wav") ||
		strings.EqualFold(filepath.Ext(path), ".wave") {
		pcm, info, err := voiceprint.ReadWAV(path)
		if err != nil {
			return nil, voiceprint.WaveInfo{}, err
		}
		want, _ := engine.SampleRate()
		if want != 0 && info.SampleRate != want {
			log.Printf("警告: %s 采样率 %d 与模型期望 %d 不一致，请先重采样",
				path, info.SampleRate, want)
		}
		return pcm, info, nil
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, voiceprint.WaveInfo{}, err
	}
	if len(raw)%4 != 0 {
		return nil, voiceprint.WaveInfo{}, fmt.Errorf("裸 PCM 长度不是 4 的倍数")
	}
	return bytesToFloats(raw), voiceprint.WaveInfo{
		SampleRate: rawRate,
		Channels:   1,
		Bits:       32,
		IsFloat:    true,
		NumSamples: len(raw) / 4,
	}, nil
}

func bytesToFloats(b []byte) []float32 {
	n := len(b) / 4
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		out[i] = voiceprint.RawFloat32LE(b[i*4:])
	}
	return out
}

func verdict(score float32) string {
	switch {
	case score >= 0.97:
		return "同一说话人（高度相似）"
	case score >= 0.90:
		return "可能同一说话人"
	case score >= 0.70:
		return "不确定"
	default:
		return "不同说话人"
	}
}
