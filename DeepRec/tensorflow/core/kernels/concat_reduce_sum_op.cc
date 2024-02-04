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

template <typename Device, typename T>
class ConcatReduceSumOp : public OpKernel {
 public:
  explicit ConcatReduceSumOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("N", &N_));
  }

  void Compute(OpKernelContext* ctx) override {
    constexpr int vec_size_avx512 = 16;
    constexpr int vecs_in_block = 4;
    constexpr int block_size_avx512 = vec_size_avx512 * vecs_in_block;
    OP_REQUIRES(ctx, N_ == 29,
            errors::InvalidArgument("Expected input number to be 29.  But received: ", 
                                    N_));

    auto batch_size = (ctx->input(0)).dim_size(0);

    TensorShape output_shape;
    OP_REQUIRES_OK(ctx, TensorShape::BuildTensorShape({batch_size, 1}, &output_shape));

    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
    // auto compute_shard = [&](int64 begin, int64 end) {
    //   for(int i = begin; i < end; ++i){
    //     T sum_value = (T)0;
    //     for(int j = 0; j < 29; j++){
    //       Tensor t = ctx->input(j);
    //       auto tmap = t.tensor<T, 2>();
    //       for(int k = 0; k < t.dim_size(1); k++){
    //         sum_value += tmap(i,k);
    //       }
    //     } 
    //     auto outmap = output->tensor<T,2>();
    //     outmap(i,0) = sum_value;
    //   }
    // };
    auto compute_shard = [&](int64 begin, int64 end) {
      for(int i = begin; i < end; ++i){
        __m512 sum_val_0 = _mm512_setzero_ps();
        __m512 sum_val_1 = _mm512_setzero_ps();
        __m512 sum_val_2 = _mm512_setzero_ps();
        __m512 sum_val_3 = _mm512_setzero_ps();
        T sum = (T)0;
        for(int j = 0; j < 29; j++){
          Tensor t = ctx->input(j);
          auto tmap = t.tensor<T, 2>();
          auto tsize = t.dim_size(1);
          if(tsize == 1){
            sum += tmap(i,0);
          }else if(tsize == 5){
            float* ptr = (float*)&tmap(i,0);
            sum_val_0 = _mm512_add_ps(sum_val_0, _mm512_maskz_loadu_ps(0x1f,ptr));
          }else{
            OP_REQUIRES(ctx, tsize == 10000,
            errors::InvalidArgument("Expected dim1 size to be 10000. But received: ", 
                                    tsize));
            int k;
            for(k = 0; k + block_size_avx512 <= tsize; k += block_size_avx512){
              float* ptr = (float*)&tmap(i,k);

              __m512 val0 = _mm512_loadu_ps(ptr + 0 * 16);
              __m512 val1 = _mm512_loadu_ps(ptr + 1 * 16);
              __m512 val2 = _mm512_loadu_ps(ptr + 2 * 16);
              __m512 val3 = _mm512_loadu_ps(ptr + 3 * 16);

              sum_val_0 = _mm512_add_ps(sum_val_0, val0);
              sum_val_1 = _mm512_add_ps(sum_val_1, val1);
              sum_val_2 = _mm512_add_ps(sum_val_2, val2);
              sum_val_3 = _mm512_add_ps(sum_val_3, val3);
            }

            // last value
            float* ptr = (float*)&tmap(i,k);
            __m512 val0 = _mm512_loadu_ps(ptr);
            sum_val_0 = _mm512_add_ps(sum_val_0, val0);

          }
        }
        //todo 手写reduce
        sum = _mm512_reduce_add_ps(_mm512_add_ps(
          _mm512_add_ps(sum_val_0, sum_val_1),
          _mm512_add_ps(sum_val_2, sum_val_3)
        ));
        auto outmap = output->tensor<T,2>();
        outmap(i,0) = sum;
      }
    };
    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
    
    Shard(worker_threads.num_threads, worker_threads.workers, 
          batch_size, 200000, compute_shard);
  }

 private:
  int32 N_;

  TF_DISALLOW_COPY_AND_ASSIGN(ConcatReduceSumOp);
};

#define REGISTER_CONCAT_REDUCE_SUM(type)                \
  REGISTER_KERNEL_BUILDER(Name("ConcatReduceSum")                        \
                              .Device(DEVICE_CPU)               \
                              .TypeConstraint<type>("T"),        \
                          ConcatReduceSumOp<CPUDevice, type>);

TF_CALL_float(REGISTER_CONCAT_REDUCE_SUM);
#undef REGISTER_CONCAT_REDUCE_SUM

}  // namespace tensorflow