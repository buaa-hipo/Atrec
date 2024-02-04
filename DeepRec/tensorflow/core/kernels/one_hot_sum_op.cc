#define EIGEN_USE_THREADS

#include <memory>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/overflow.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T, typename TI>
class OneHotSumOp : public OpKernel {
 public:
  explicit OneHotSumOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("axis", &axis_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& indices = ctx->input(0);
    const Tensor& depth = ctx->input(1);
    const Tensor& on_value = ctx->input(2);
    const Tensor& off_value = ctx->input(3);
    const TensorShape& indices_shape = indices.shape();

    const int indices_dims = indices_shape.dims();
    const int output_dims = indices_dims;

    // Preliminary validation of sizes.
    OP_REQUIRES(ctx, indices_dims >= 1,
        		    errors::InvalidArgument("Expected indices dims to be no less than 1.  "
										                    "But received: ", indices_dims));

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(depth.shape()),
                errors::InvalidArgument("depth must be a scalar, but got: ",
                                        depth.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(on_value.shape()),
                errors::InvalidArgument("on_value must be a scalar, but got: ",
                                        on_value.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(off_value.shape()),
                errors::InvalidArgument("off_value must be a scalar, but got: ",
                                        off_value.shape().DebugString()));

    /// off_value must be zero because it would not be added at all
    OP_REQUIRES(ctx, off_value.scalar<T>()() == T(0.0),
                errors::InvalidArgument("off_value must be zero, but got: ",
                                        off_value.scalar<T>()()));

    /// off_value must be zero because it would not be added at all
    OP_REQUIRES(ctx, off_value.scalar<T>()() == T(0.0),
                errors::InvalidArgument("off_value must be zero, but got: ",
                                        off_value.scalar<T>()()));
    // only the last axis is supported
    OP_REQUIRES(ctx, axis_ == -1,
                errors::InvalidArgument("Expected axis to be -1.  But received: ", 
                                        axis_));

    const int axis = indices_dims - 1;

    // The one-hot dimension.
    const int32 depth_v = depth.scalar<int32>()();
    OP_REQUIRES(
        ctx, depth_v >= 0,
        errors::InvalidArgument("depth must be non-negative, got: ", depth_v));
    OP_REQUIRES(
        ctx,
        MultiplyWithoutOverflow(indices_shape.num_elements(), depth_v) >= 0,
        errors::InvalidArgument("OneHotSum result would have shape ",
                                indices_shape.DebugString(), " + [", depth_v,
                                "], which exceeds 2**63 - 1 elements"));

    TensorShape output_shape = indices_shape;
    output_shape.set_dim(axis, depth_v);

    auto on_value_t = on_value.scalar<T>();
    auto off_value_t = off_value.scalar<T>();

    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    if (output_shape.num_elements() > 0) {
      // prefix_dim_size == # of elements before the axis
      // depth_v == # of elements per axis
      int64 prefix_dim_size = indices_shape.num_elements() / indices_shape.dim_size(axis);
      int64 sum_dim_size = indices_shape.dim_size(axis);

      // Reshape indices into 1-d array of size prefix_dim_size
      auto indices_t = indices.shaped<TI, 2>({prefix_dim_size, sum_dim_size});
      // Split output into 2-Tensor of size:
      //   prefix_dim_size x depth
      auto output_t = output->shaped<T, 2>({prefix_dim_size, depth_v});

      auto compute_shard = [&](int64 begin, int64 end) {
        for (int i = begin; i < end; ++i) {
          for (int j = 0; j < depth_v; j++) {
            output_t(i, j) = off_value_t();
          }
          for (int j = 0; j < sum_dim_size; j++) {
            // /Tianchi bzdjsm/ cnmd，one hot的文档里怎么从没写过-1怎么处理啊？还得上stack overflow上搜
            auto index = indices_t(i, j);
            if (index >= 0) {
              output_t(i, indices_t(i, j)) += on_value_t();
            }
          }
        }
      };

      auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());

      Shard(worker_threads.num_threads, worker_threads.workers, 
            prefix_dim_size, 20 * sum_dim_size, compute_shard);
    }

  }

 private:
  int32 axis_;

  TF_DISALLOW_COPY_AND_ASSIGN(OneHotSumOp);
};

#define REGISTER_ONE_HOT_SUM_INDEX(type, index_type)                \
  REGISTER_KERNEL_BUILDER(Name("OneHotSum")                        \
                              .Device(DEVICE_CPU)               \
                              .TypeConstraint<index_type>("TI") \
                              .TypeConstraint<type>("T")        \
                              .HostMemory("depth"),             \
                          OneHotSumOp<CPUDevice, type, index_type>);

#define REGISTER_ONE_HOT_SUM(type)         \
  REGISTER_ONE_HOT_SUM_INDEX(type, uint8); \
  REGISTER_ONE_HOT_SUM_INDEX(type, int32); \
  REGISTER_ONE_HOT_SUM_INDEX(type, int64)

TF_CALL_NUMBER_TYPES(REGISTER_ONE_HOT_SUM);
#undef REGISTER_ONE_HOT_SUM
#undef REGISTER_ONE_HOT_SUM_INDEX

}  // namespace tensorflow