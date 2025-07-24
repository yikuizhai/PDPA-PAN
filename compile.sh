#!/bin/bash
export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9
export TORCH_CUDA_ARCH_LIST="7.5"  # 使用兼容架构
pip install -e .
