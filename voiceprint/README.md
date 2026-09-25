# voiceprint — Go 声纹特征库绑定

对 C 声纹特征库 `libvoiceprint` 的 cgo 封装，提供：

| 能力 | Go API |
|------|--------|
| 加载模型 | `voiceprint.Open(modelPath string)` |
| 提取特征 | `(*Engine).Extract(pcm []float32) (Embedding, error)` |
| 比较相似度 | `(*Engine).Compare(a, b Embedding) (float32, error)` |
| 音频直比 | `(*Engine).CompareAudio(pcmA, pcmB []float32) (float32, error)` |
| 释放资源 | `(*Engine).Close()`（可重复调用，另设 GC finalizer 兜底） |
| 错误文字 | `*voiceprint.Error{Code, Msg}`，实现了 `error`，可用 `errors.As` |
| 日志回调 | `(*Engine).SetLogger(func(level int, msg string))` |

`cmd/voiceprint` 是示例命令：输入两段音频，输出相似分。

## 依赖

- cgo（需要 C 编译器，如 gcc/clang）
- 链接数学库 `-lm`（已在源码中用 `#cgo LDFLAGS: -lm` 声明）

```sh
export CGO_ENABLED=1
go test ./...
go build ./cmd/voiceprint
```

## 命令行用法

```sh
# 内置默认模型
voiceprint a.wav b.wav

# 指定模型、打开底层日志
voiceprint -model path/to/model.vpm -v a.wav b.wav
```

输出示例：

```
音频1: a1.wav (16000 Hz)
音频2: a2.wav (16000 Hz)
特征维度: 64
声纹相似度: 0.998347
判定: 同一说话人（高度相似）
```

- `.wav/.wave`：支持 8/16/24/32 位整数 PCM、32 位 float WAV，自动混为单声道
- 其它扩展名：按原始 `float32 LE` 单声道 PCM 读取，`-raw-rate` 指定采样率
- 不会自动重采样，请保证采样率与模型期望一致（默认 16000 Hz）

## 代码中最小用法

```go
package main

import (
    "fmt"
    "log"

    "voiceprint"
)

func main() {
    eng, err := voiceprint.Open("") // 空串 = 内置默认模型
    if err != nil {
        log.Fatal(err)
    }
    defer eng.Close()

    // pcm 为 []float32、单声道、16kHz、取值 [-1,1]
    ea, err := eng.Extract(pcmA)
    if err != nil { log.Fatal(err) }
    eb, err := eng.Extract(pcmB)
    if err != nil { log.Fatal(err) }

    score, err := eng.Compare(ea, eb) // [-1,1]，越接近 1 越像同一人
    if err != nil { log.Fatal(err) }
    fmt.Println(score)
}
```

错误处理：

```go
_, err := eng.Extract(pcm)
var ve *voiceprint.Error
if errors.As(err, &ve) {
    fmt.Println("底层错误码:", ve.Code, "文字:", ve.Msg)
}
```

## 内存与回调的所有权约定（不长期持有 Go 内存）

底层通过 cgo 与 Go 交互，本绑定遵循以下规则，避免 Go 内存被底层长期引用：

1. **特征缓冲：复制后立即释放。**
   `vp_embed` 返回的 C 缓冲在 Go 侧 `copy` 成普通 `[]float32` 后，
   立刻通过 `defer vp_free(...)` 释放。Go 拿到的 `Embedding` 不依赖 C 内存。

2. **入参缓冲：调用期临时分配，返回即释放。**
   传入 C 的音频/特征使用 `malloc + copy` 得到的临时 C 缓冲
   （`floatsToC`），函数返回前 `free`；并对原始 Go 切片加
   `runtime.KeepAlive` 防止提前回收。不把 Go 指针直接传给 C 存储。

3. **日志回调：只传整数句柄，不传 Go 指针。**
   Go 闭包用 `runtime/cgo.Handle`（一个 uintptr 值）登记；
   C 侧保存的是代码段中的**静态跳板函数指针** + 这个整数
   （见 `cgo_bridge.c`），不包含 Go 堆指针。回调时跳板用句柄取回闭包，
   `C.GoString` 当场把消息复制成 Go string；`Close`/替换回调时先摘除 C
   回调再 `handle.Delete()`，C 不可能再用到失效句柄。

4. **资源释放有顺序保证。**
   `Close` 先调用 `vp_destroy`（其返回后底层不再触碰回调和缓冲），
   然后才删除 cgo 句柄；`closeOnce` 保证幂等。忘记 `Close` 时，
   `runtime.SetFinalizer` 兜底释放引擎。

> 注意：`Logger` 在 cgo 调用栈上同步执行，回调内不要再次调用同一
> `Engine` 的方法（方法用互斥锁串行化，重入会自锁），也不要长期保存
> 回调参数（Go 字符串赋值即拷贝，可放心保存拷贝）。

## 文件说明

```
voiceprint.h        底层 C 接口（ABI 头文件）
voiceprint.c        参考实现（STFT 对数谱 + 子带重心，64 维；可替换成真实引擎）
cgo_bridge.c        Go 日志回调的 C 跳板与句柄包装
voiceprint.go       cgo 绑定：Open/Extract/Compare/Close/SetLogger/Error
wav.go              WAV 读取与 PCM->float32 转换
cmd/voiceprint/     示例命令
voiceprint_test.go  单元/并发/ASan 测试
```

测试已通过 `go test -race` 以及 AddressSanitizer/LeakSanitizer。
