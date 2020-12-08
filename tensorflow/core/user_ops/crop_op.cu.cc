#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "crop_op.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

// Define the CUDA kernel.
// template 
// __global__ void CropCudaKernel
// ...
// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
void CropFunctor<GPUDevice, T>::operator()(
    const GPUDevice& d,
    const T* image_ptr,
    const int* crop_centers_ptr,
    int crop_size,
    int image_size,
    int channels,
    int num_crops,
    T* crops_ptr
  ) {
  // Launch the cuda kernel.
  int block_count = num_crops;
  int thread_per_block = 1024;
  CropCudaKernel<T>
      <<<block_count, thread_per_block, 0, d.stream()>>>(
        image_ptr,
        crop_centers_ptr,
        image_size,
        channels,
        crop_size,
        num_crops,
        crops_ptr
      );
    cudaDeviceSynchronize();
}

// Explicitly instantiate functors for the types of OpKernels registered.
template struct CropFunctor<GPUDevice, float>;
template struct CropFunctor<GPUDevice, int32>;

#endif // GOOGLE_CUDA