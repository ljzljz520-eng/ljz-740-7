/*
 * cgo_bridge.c — C 回调到 Go 的桥接层。
 *
 * 为什么需要这一层：
 *  1) 使用 //export 时，cgo 前言里只能放声明，跳板的“定义”必须在独立 C 文件；
 *  2) runtime/cgo 的 Handle 是 uintptr 值。让 Go 侧直接把整数传给
 *     vp_bridge_set_logger，可避免 unsafe.Pointer(uintptr) 转换，
 *     也满足 go vet 的 cgo 指针规则；
 *  3) C 库最终拿到的只是代码段中的静态函数指针和一个不透明数值，
 *     不持有任何 Go 堆指针。
 */
#include "voiceprint.h"
#include <stdint.h>

/* Go 通过 //export 导出（生成的原型不带 const）。 */
extern void goVPLogCallback(int level, char *msg, uintptr_t user_data);

static void vp_go_log_trampoline(int level, const char *msg, void *user_data) {
	goVPLogCallback(level, (char *)msg, (uintptr_t)(void *)user_data);
}

/* handle==0 表示清除回调。 */
void vp_bridge_set_logger(vp_engine *e, uintptr_t handle) {
	if (handle == 0) {
		vp_set_logger(e, 0, 0);
	} else {
		vp_set_logger(e, &vp_go_log_trampoline, (void *)(uintptr_t)handle);
	}
}
