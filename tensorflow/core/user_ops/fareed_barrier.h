// fareed_barrier.h
#ifndef FAREED_BARRIER_H_
#define FAREED_BARRIER_H_

#include "tensorflow/core/util/gpu_kernel_helper.h"

using namespace tensorflow;

template <typename Device, typename T>
struct ExampleFunctor {
  void operator()(const Device& d, int size, const T* in, T* out);
};

#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename T>
struct ExampleFunctor<Eigen::GpuDevice, T> {
  void operator()(const Eigen::GpuDevice& d, int size, const T* in, T* out);
};
#endif

#endif FAREED_BARRIER_H_