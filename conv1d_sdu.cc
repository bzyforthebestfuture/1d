#include "tensorflow/lite/kernels/conv_1d_sdu.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <vector>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace custom {
namespace conv_1d {
    //------------------------------------------------------------------------0
const int dim = 5;  // 滑动窗口的大小
//    float copy[dim];  // 临时存储滑动窗口的数据
// constexpr int dilation = 2;  // 膨胀因子

int dim_in;  // 输入数据的维度
int dim_out;  // 输出数据的维度
const int kernal_size;  // 卷积核的大小  dim_k
//constexpr float kernel[dim_k] = {1.2, 2.0, 4.2};  // 卷积核???

TfLiteStatus Conv1dPrepare(TfLiteContext* context, TfLiteNode* node) 
{
    //确保数据有效性
    TF_LITE_ENSURE_EQ(context, NumInputs(node), 1); 
    TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1); 

    // 获取张量
    const TfLiteTensor* input = GetInput(context, node, 0); 
    TfLiteTensor* output = GetOutput(context, node, 0);  

    int num_dims = NumDimensions(input);  // 获取输入张量的维度数量 (1250,1,1,1) 有4个维度
                //------------------------------------------------------------------------1
    // 创建并设置输出张量的尺寸

    //<设置为3？还是4？
    TfLiteIntArray* output_size = TfLiteIntArrayCreate(num_dims);//设置输出维度
    for (int i = 0; i < num_dims; ++i) {
        output_size->data[i] = input->dims->data[i];
    }

    return context->ResizeTensor(context, output, output_size);
}
// 计算函数，执行1D卷积操作
TfLiteStatus Conv1dEval(TfLiteContext* context, TfLiteNode* node)
{
    const TfLiteTensor* input = GetInput(context, node, 0);  // 获取输入张量
    TfLiteTensor* output = GetOutput(context, node, 0);  // 获取输出张量

    float* input_data = input->data.f;  // 获取输入数据
    float* output_data = output->data.f;  // 获取输出数据

    // 获取输入和输出的维度
    if (output->dims->data[0] > 1)
        dim_out = output->dims->data[0];
    else
        dim_out = output->dims->data[1];

    if (input->dims->data[0] > 1)
        dim_in = input->dims->data[0];
    else
        dim_in = input->dims->data[1];
    
      //------------------------------------------------------------------------2
    // // 初始化前四个元素为0
    // for (int i = 0; i < 4; i++) {
    //     copy0[i] = 0;
    // }

    // // 将输入数据复制到扩展的存储中
    // for (int i = 0; i < dim_in; i++) {
    //     copy0[i + 4] = input_data[i];
    // }

    // // 滑动窗口进行卷积操作
    // for (int i = 0; i < dim_out; i++) {
    //     for (int m = 0; m < dim; m++) {
    //     copy[m] = copy0[m + i];
    //     }
    //     for (int j = 0; j < dim_k; j++) {
    //     output_data[i] = output_data[i] + copy[j * dilation] * kernel[j];
    //     }
    // }
    
    // 滑动窗口进行卷积操作
for (int i = 0; i < dim_out; i++) {
    for (int m = 0; m < dim; m++) {//------------------------------------3
        copy[m] = input_data[m + i];
    }
    for (int j = 0; j < dim_k; j++) {
        output_data[i] = output_data[i] + copy[j * dilation] * kernel[j];
    }
}

}

}
// 注册Conv1d操作
TfLiteRegistration* Register_CONV_1D() {
  static TfLiteRegistration r = {nullptr, nullptr, conv_1d::Conv1dPrepare, conv_1d::Conv1dEval};
  return &r;
}
}  // namespace custom
}  // namespace ops
}  // namespace tflite    