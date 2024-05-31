/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.



Licensed under the Apache License, Version 2.0 (the "License");

you may not use this file except in compliance with the License.

You may obtain a copy of the License at



    http://www.apache.org/licenses/LICENSE-2.0



Unless required by applicable law or agreed to in writing, software

distributed under the License is distributed on an "AS IS" BASIS,

WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and

limitations under the License.

==============================================================================*/

//在af_detection包中添加

#include "tensorflow/lite/micro/examples/person_detection/main_functions.h"

#include "tensorflow/lite/interpreter.h"

#include "tensorflow/lite/kernels/register.h"

#include "tensorflow/lite/model.h"

#include "tensorflow/lite/kernels/conv1d_sdu.h"

// This is the default main used on systems that have the standard C entry

// point. Other devices (for example FreeRTOS or ESP32) that have different

// requirements for entry code (like an app_main function) should specialize

// this main.cc file in a target-specific subfolder.

int main(int argc, char* argv[]) {
//new----------------------------------------------------------------------------------------------------------------------
// 上传tflite模型
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile("afnet.tflite");
    if (!model) {
        fprintf(stderr, "Failed to load model\n");

        return 1;

    }
    // 建立算子到函数的关联
    tflite::ops::builtin::BuiltinOpResolver resolver;
    resolver.AddCustom("Conv1D_SDU", tflite::ops::custom::conv1d_sdu::Register_CONV_1D_SDU());
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        fprintf(stderr, "Failed to construct interpreter\n");
        return 1;
    }
    // Allocate tensor buffers
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        fprintf(stderr, "Failed to allocate tensors\n");
        return 1;
    }
//new----------------------------------------------------------------------------------------------------------------------
  setup();
  int n=10;
  while (n--) {
    loop();
  }
}