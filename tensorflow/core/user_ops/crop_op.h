#ifndef CROP_OP_H_
#define CROP_OP_H_

#include "tensorflow/core/util/gpu_kernel_helper.h"

using namespace tensorflow;

template <typename Device, typename T>
struct CropFunctor {
  void operator()(
    const Device& d,
    const T* image_ptr,
    const int* crop_centers_ptr,
    int crop_size,
    int image_size,
    int channels,
    int num_crops,
    T* crops_ptr
  );
};

#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename T>
struct CropFunctor<Eigen::GpuDevice, T> {
  void operator()(
    const Eigen::GpuDevice& d,
    const T* image_ptr,
    const int* crop_centers_ptr,
    int crop_size,
    int image_size,
    int channels,
    int num_crops,
    T* crops_ptr
  );
};
#endif

#endif // CROP_OP_H_