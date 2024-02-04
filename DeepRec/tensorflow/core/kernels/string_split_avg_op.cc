/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/string_ops.cc.

#include <errno.h>
#include <immintrin.h>
#include <stdlib.h>

#include <algorithm>
#include <numeric>
#include <string>

#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
using thread::ThreadPool;

// Split input string `str` based on a character delimiter and divide by
// length of items
template <typename OutputType>
class StringSplitAvgOp : public OpKernel {
 public:
  explicit StringSplitAvgOp(OpKernelConstruction* context)
      : OpKernel(context), maxsplit_(-1) {
    OP_REQUIRES_OK(context, context->GetAttr("maxsplit", &maxsplit_));
  }
  using OpKernel::OpKernel;

  void ParseNumbers(
      std::vector<float>& values,
      const Eigen::TensorMap<Eigen::Tensor<const string, 1, 1, long>, 16,
                             Eigen::MakePointer>& input_vec,
      const int64& batch_index, const size_t& start_index, const size_t& len) {
    StringPiece text(input_vec(batch_index).data() + start_index, len);
    float temp = 0.;
    strings::SafeStringToNumeric<float>(text, &temp);
    values.emplace_back(temp);
  }

  int ParseNumbers(
      const Eigen::TensorMap<Eigen::Tensor<const string, 1, 1, long>, 16,
                             Eigen::MakePointer>& input_vec,
      const int64& batch_idx, const size_t& start_index) {
    char* p = (char*)(input_vec(batch_idx).data() + start_index);
    int flag = 0, c = 0, d = 0;
    if (*(p + c) == '-')
      flag = 1, c++;
    else if (*(p + c) == '+')
      c++;

    for (; (*(p + c) >= '0') && (*(p + c) <= '9'); c++)
      d = (d << 3) + (d << 1) + (*(p + c) - '0');

    if (*(p + c) == '.') {
      c++;
    
      if ((*(p + c) >= '0') && (*(p + c) <= '9'))
        d = (d << 3) + (d << 1) + (*(p + c) - '0');
      else
        d = (d << 3) + (d << 1);
      c++;

      if ((*(p + c) >= '0') && (*(p + c) <= '9'))
        d = (d << 3) + (d << 1) + (*(p + c) - '0');
      else
        d = (d << 3) + (d << 1);
      c++;
      
      if ((*(p + c) >= '5') && (*(p + c) <= '9')) d++;
    } else {
      d = (d << 6) + (d << 5) + (d << 2);
    }

    return (flag) ? 0 - d : d;
  }

  void Compute(OpKernelContext* context) override {
    // This is not a deep copy of the input tensor; they will share the same
    // underlying storage.
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_tensor->shape()),
                errors::InvalidArgument("input must be a vector, got shape: ",
                                        input_tensor->shape().DebugString()));

    const auto input_vec = input_tensor->vec<string>();
    const int64 batch_size = input_vec.dimension(0);

    const Tensor* sep_tensor;
    OP_REQUIRES_OK(context, context->input("sep", &sep_tensor));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(sep_tensor->shape()),
                errors::InvalidArgument("sep must be a scalar, got shape: ",
                                        sep_tensor->shape().DebugString()));
    const auto sep_vec = sep_tensor->flat<string>();

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       "output", TensorShape({batch_size, 1}), &output_tensor));
    auto output_vec = output_tensor->flat<OutputType>();

#if defined(__AVX512F__)
    auto RunTask = [this, &input_vec, &sep_vec, &output_vec](int64 start,
                                                             int64 end) {
      if (sep_vec(0).size() == 1) {
        char sep = *(sep_vec(0).data());
        __m512i sep512 = _mm512_set1_epi8(sep);
        for (int64 batch_idx = start; batch_idx < end; batch_idx++) {
          __mmask64 mask = 0;
          __mmask64 temp_mask = 0;
          size_t remain_size = 0;
          size_t seq_idx = 0;
          size_t offset = 0;
          int sum = 0;
          size_t cnt = 0;
          int temp = 0;
          __m512i x;
          int64 part_cnt;
          for (; seq_idx + 63 < input_vec(batch_idx).size(); seq_idx += 64) {
            x = _mm512_loadu_si512(input_vec(batch_idx).data() + seq_idx);
            mask = _mm512_cmpeq_epi8_mask(x, sep512);
            part_cnt = _mm_popcnt_u64(_cvtmask64_u64(mask));
            cnt += part_cnt;
            temp_mask = mask;
            for (int64 k = 0; k < part_cnt; k++) {
              int64 lz = (int64)_tzcnt_u64(_cvtmask64_u64(temp_mask));
              int64 len = k == 0 ? lz + remain_size : lz;
              sum += ParseNumbers(input_vec, batch_idx, offset);
              temp_mask = temp_mask >> (lz + 1);
              offset += (len + 1);
            }
            remain_size = (int64)_lzcnt_u64(_cvtmask64_u64(mask));
          }
          x = _mm512_mask_loadu_epi8(
              _mm512_setzero_si512(),
              0xFFFFFFFFFFFFFFFF >> (seq_idx + 63 - input_vec(batch_idx).size()),
              input_vec(batch_idx).data() + seq_idx);
          mask = _mm512_cmpeq_epi8_mask(x, sep512);
          temp_mask = mask;
          part_cnt = _mm_popcnt_u64(_cvtmask64_u64(mask));
          cnt += (part_cnt + 1);
          for (int64 k = 0; k < part_cnt; k++) {
            int64 lz = (int64)_tzcnt_u64(_cvtmask64_u64(temp_mask));
            int64 len = (k == 0 ? lz + remain_size : lz);
            sum += ParseNumbers(input_vec, batch_idx, offset);
            temp_mask = temp_mask >> (lz + 1);
            offset += (len + 1);
          }
          sum+= ParseNumbers(input_vec, batch_idx, offset);
          output_vec(batch_idx) = static_cast<OutputType>(sum / ((1e-7 + cnt) * 100));
        }
      }
    };
    if (false){ // isParallel
      auto worker_threads = context->device()->tensorflow_cpu_worker_threads();
      Shard(worker_threads->num_threads - 1, worker_threads->workers,
            input_vec.size(), 25, RunTask);
    } else{
      RunTask(0, batch_size);
    }
#endif
  }

 private:
  int maxsplit_;
};

// Registers the currently supported output types.
#define REGISTER(type)                                           \
  REGISTER_KERNEL_BUILDER(Name("StringSplitAvg")                 \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<type>("out_type"), \
                          StringSplitAvgOp<type>)
REGISTER(float);
REGISTER(double);
REGISTER(bfloat16);
#undef REGISTER

}  // namespace tensorflow
