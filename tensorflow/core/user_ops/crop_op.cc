#include "crop_op.h"

using namespace tensorflow;

// Register TF operation
REGISTER_OP("Crop")
    .Attr("T: {float, int32} = DT_FLOAT")
    .Input("image: float32")
    .Input("crop_centers: int32")
    .Input("crop_size: int32")
    .Output("crops: float32");

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// CPU specialization of actual computation.
template <typename T>
struct CropFunctor<CPUDevice, T> {
  void operator()(
    const CPUDevice& d,
    const T* image_ptr,
    const int* crop_centers_ptr,
    int crop_size,
    int image_size,
    int channels,
    int num_crops,
    T* crops_ptr
  ) {

  }
};

// OpKernel definition.
// template parameter  is the datatype of the tensors.
template <typename Device, typename T>
class CropOp : public OpKernel {
 public:
  explicit CropOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensors
    const Tensor& image = context->input(0);
    const Tensor& crop_centers = context->input(1);
    const Tensor& crop_size_tensor = context->input(2);
    // FIXME
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(crop_size_tensor.shape()), errors::InvalidArgument("crop_size must be scalar, has shape ", crop_size_tensor.shape().DebugString()));
    //const int crop_size = crop_size_tensor.scalar()();
    const int crop_size = 64;
    // Get shapes of input tensors
    const TensorShape& image_shape = image.shape();
    const TensorShape& crop_centers_shape = crop_centers.shape();
    int image_size = image_shape.dim_size(1);
    int channels = image_shape.dim_size(3);
    int num_crops = crop_centers_shape.dim_size(0);
    int dim = crop_centers_shape.dim_size(1);

    // Create an output tensor
    Tensor* crops = NULL;
    // create output shape
    TensorShape crops_shape;
    crops_shape.AddDim(num_crops);
    crops_shape.AddDim(crop_size);
    crops_shape.AddDim(crop_size);
    crops_shape.AddDim(crop_size);
    crops_shape.AddDim(channels);
    OP_REQUIRES_OK(context, context->allocate_output(0, crops_shape,
                                                     &crops));
                                                     // Do the computation.
    CropFunctor<Device, T>()(
        context->eigen_device<Device>(),
        image.flat<T>().data(),
        crop_centers.flat<int>().data(),
        crop_size,
        image_size,
        channels,
        num_crops,
        crops->flat<T>().data()
      );

  }
};

// Register the CPU kernels.
#define REGISTER_CPU(T) \
  REGISTER_KERNEL_BUILDER( \
      Name("Crop") \
      .Device(DEVICE_CPU) \
      .TypeConstraint("T"), \
    CropOp);
REGISTER_CPU(float);
REGISTER_CPU(int32);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T) \
  extern template struct CropFunctor; \
  REGISTER_KERNEL_BUILDER( \
      Name("Crop")      \
      .Device(DEVICE_GPU)   \
      .TypeConstraint("T"),  \
    CropOp);
REGISTER_GPU(float);
REGISTER_GPU(int32);
#endif // GOOGLE_CUDA