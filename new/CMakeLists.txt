cmake_minimum_required(VERSION 3.10)

# 项目信息
project(conv1d)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
include_directories(

    /home/fjjt/Downloads/iesdcontest2024_demo_example_evaluation/tensorflow/lite/c

    /home/fjjt/Downloads/iesdcontest2024_demo_example_evaluation/tensorflow/lite/kernels/internal

    # Add other directories if necessary

)



# 包含TensorFlow Lite头文件

include_directories(/home/fjjt/Downloads/iesdcontest2024_demo_example_evaluation/tensorflow)

include_directories(/home/fjjt/Downloads/iesdcontest2024_demo_example_evaluation/tensorflow/lite)



# 添加共享库

add_library(conv1d SHARED conv1d.cc)



# 链接TensorFlow Lite库

target_link_libraries(conv1d /home/fjjt/Downloads/iesdcontest2024_demo_example_evaluation/tensorflow/lite/build/libtensorflow-lite.a)

