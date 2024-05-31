#ifndef TENSORFLOW_LITE_KERNELS_CONV1D_SDU_H_
#define TENSORFLOW_LITE_KERNELS_CONV1D_SDU_H_

#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace ops {
namespace custom {
namespace conv_1d {

// 准备函数的声明
TfLiteStatus Conv1dPrepare(TfLiteContext* context, TfLiteNode* node);

// 计算函数的声明
TfLiteStatus Conv1dEval(TfLiteContext* context, TfLiteNode* node);

// 自定义操作注册函数的声明
TfLiteRegistration* Register_CONV_1D();

}  // namespace conv_1d
}  // namespace custom
}  // namespace ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_CONV1D_SDU_H_
