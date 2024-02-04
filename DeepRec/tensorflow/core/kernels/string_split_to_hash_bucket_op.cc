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
#include <algorithm>
#include <numeric>
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/lib/hash/hash.h"

namespace tensorflow {

using thread::ThreadPool;
namespace {
// Split input string `str` based on a character delimiter.
// Returns a vector of StringPieces which are valid as long as input `str`
// is valid.
// Note: The single character delimiter is a common case and is implemented as
// a series of finds in the input string, making it much more effcient than
// SplitOnCharSet.

class WorkerInfo{
 public:
  explicit WorkerInfo(int num_threads, int64 batch_size)
      :output_size(0), counter_for_thread(0),
      thread_index(0), max_num_entries(0) {
      const int kReserveSize = 4;
      int numReserve = batch_size * kReserveSize / num_threads ?
                       batch_size * kReserveSize / num_threads : 1;
      tokens_buffer.reserve(numReserve);
      num_indices_buffer.assign(batch_size, 0);
  }

  std::vector<int64> tokens_buffer;
  int64 output_size;
  int64 counter_for_thread;
  int64 thread_index;
  int64 max_num_entries;
  std::vector<int64> num_indices_buffer;
};

int64 HashBucket(StringPiece* token, int64 num_buckets) {
  const uint64 input_hash = Hash64(token->data(), token->size());
  if (num_buckets > 0) {
    const uint64 bucket_id = input_hash % num_buckets;
    return static_cast<int64>(bucket_id);
  } else {
    return static_cast<int64>(input_hash);
  }
}

std::vector<int64> SplitV2(const string& str, StringPiece sep, int64 num_buckets, int maxsplit) {
  // This SplitV2 method matches the behavior of python's str.split:
  //   If sep is given, consecutive delimiters are not grouped together
  //   and are deemed to delimit empty strings (for example, '1,,2'.split(',')
  //   returns ['1', '', '2']). The sep argument may consist of multiple
  //   characters (for example, '1<>2<>3'.split('<>') returns ['1', '2', '3']).
  //   Splitting an empty string with a specified separator returns [''].
  //
  //   If sep is not specified or is None, a different splitting algorithm is
  //   applied: runs of consecutive whitespace are regarded as a single
  //   separator, and the result will contain no empty strings at the start or
  //   end if the string has leading or trailing whitespace. Consequently,
  //   splitting an empty string or a string consisting of just whitespace
  //   with a None separator returns [].

  std::vector<int64> result;
  StringPiece text(str);
  if (maxsplit == 0) {
    return result;
  }

  if (sep.empty()) {
    StringPiece token;
    // Remove leading whitespaces.
    str_util::RemoveLeadingWhitespace(&text);
    int split = 0;
    while (str_util::ConsumeNonWhitespace(&text, &token)) {
      result.push_back(HashBucket(&token, num_buckets));
      str_util::RemoveLeadingWhitespace(&text);
      ++split;
      if (maxsplit > 0 && split == maxsplit) {
        return result;
      }
    }
    return result;
  }
  auto p = std::search(text.begin(), text.end(), sep.begin(), sep.end());
  int split = 0;
  while (p != text.end()) {
    StringPiece token = text.substr(0, p - text.begin());
    result.push_back(HashBucket(&token, num_buckets));
    text.remove_prefix(token.size());
    text.remove_prefix(sep.size());
    ++split;
    if (maxsplit > 0 && split == maxsplit) {
      // result.push_back(text);
      return result;
    }
    p = std::search(text.begin(), text.end(), sep.begin(), sep.end());
  }
  result.push_back(HashBucket(&text, num_buckets));
  return result;
}

}  // namespace


class StringSplitToHashBucketOp : public OpKernel {
 public:
  explicit StringSplitToHashBucketOp(OpKernelConstruction* context)
      : OpKernel(context), maxsplit_(-1), element_cost_(0), result_cost_(0), num_buckets_(-1) {
    OP_REQUIRES_OK(context, context->GetAttr("num_buckets", &num_buckets_));
    OP_REQUIRES_OK(context, context->GetAttr("maxsplit", &maxsplit_));
  }

  void BatchParallelCompute(
      OpKernelContext* ctx,
      const Eigen::TensorMap<Eigen::Tensor<const string, 1, 1, long>, 16, Eigen::MakePointer> input_vec,
      const int64 batch_size,
      StringPiece sep) {
    ThreadPool* thread_pool = ctx->device()->tensorflow_cpu_worker_threads()->workers;
    const int64 num_threads = thread_pool->NumThreads() + 1;

    // 记录 batch 内每个样本对应的 indices 个数
    std::vector<int64> num_indices(batch_size);
    num_indices[0] = 0;

    std::vector<WorkerInfo> worker_info_array;
    for (int i = 0; i < num_threads; i++) {
      WorkerInfo w(num_threads, batch_size);
      worker_info_array.emplace_back(w);
    }

    std::vector<std::vector<int64>> id_to_worker(batch_size);
    int64 num_buckets = num_buckets_;
    auto RunTask = [&sep, &worker_info_array, &id_to_worker,
      &input_vec, &ctx, this,
      &num_indices, num_buckets](int64 start, int64 end, int64 worker_id) {
      // 获取当前 worker 信息
      WorkerInfo* worker_info = &worker_info_array[worker_id];
      // 遍历当前区间内的样本
      for (int64 i = start; i < end; ++i) {
        std::vector<int64> parts = SplitV2(input_vec(i), sep, num_buckets, maxsplit_);
        // 记录样本 i 的起始索引位于哪个 worker 的第几个位置, 及其长度
        id_to_worker[i].emplace_back(worker_id);
        id_to_worker[i].emplace_back(worker_info->counter_for_thread);
        id_to_worker[i].emplace_back(worker_info->output_size);

        int64 n_entries = parts.size();
        num_indices[i] = n_entries;  // 当前样本 token 个数
        worker_info->num_indices_buffer[worker_info->counter_for_thread] = n_entries;
        worker_info->output_size += n_entries;
        worker_info->max_num_entries = std::max(worker_info->max_num_entries, n_entries);
        worker_info->tokens_buffer.insert(
          worker_info->tokens_buffer.end(),
          std::make_move_iterator(parts.begin()),
          std::make_move_iterator(parts.end()));
        worker_info->counter_for_thread++;
      }
    };

    thread_pool->ParallelForWithWorkerId(batch_size, element_cost_, RunTask);

    // 将多个 worker 的信息合并, 得到所有样本总 token 数 (用于分配 indices 和 values 内存)
    // 以及样本内最多 token 数 (用于确定 shape)
    int64 output_size = 0;
    int64 max_num_entries = 0;
    for (int i = 0; i < num_threads; i++) {
      output_size += worker_info_array[i].output_size;
      max_num_entries = std::max(worker_info_array[i].max_num_entries, max_num_entries);
    }

    std::vector<int64> id_to_index(batch_size);
    id_to_index[0] = 0;
    for (size_t i = 1; i < batch_size; i++)
      id_to_index[i] = id_to_index[i-1] + num_indices[i-1];

    Tensor* sp_indices_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({output_size, 2}), &sp_indices_t));
    Tensor* sp_tokens_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape({output_size}), &sp_tokens_t));
    Tensor* sp_shape_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, TensorShape({2}), &sp_shape_t));

    auto sp_indices = sp_indices_t->matrix<int64>();
    auto sp_tokens = sp_tokens_t->vec<int64>();
    auto sp_shape = sp_shape_t->vec<int64>();
    sp_shape(0) = batch_size;
    sp_shape(1) = max_num_entries;

    auto RunTask1 = [&id_to_index, batch_size, &worker_info_array,
      &id_to_worker, &sp_indices, &num_indices,
      &sp_tokens, ctx](int64 start, int64 end, int64 worker_id) {
      for (int64 i = start; i < end; i++) {
        int last_worker_id = id_to_worker[i][0];
        int64 id_in_last_worker = id_to_worker[i][1];
        int64 st = id_to_worker[i][2];
        for (int64 j = 0; j < worker_info_array[last_worker_id].num_indices_buffer[id_in_last_worker]; j++) {
            size_t c = id_to_index[i] + j;
            sp_indices(c, 0) = i;
            sp_indices(c, 1) = j;
            sp_tokens(c) = worker_info_array[last_worker_id].tokens_buffer[st+j];
        }
      }
    };

    if (result_cost_ == 0) {
      uint64 start = 0;
      uint64 end = 0;
      size_t sample_id = rand() % batch_size;
      start = Env::Default()->NowNanos();
      RunTask1(sample_id, sample_id+1, 0);
      end = Env::Default()->NowNanos();
      result_cost_ = end - start;
    }
    uint64 result_cost = result_cost_;
    thread_pool->ParallelForWithWorkerId(batch_size, result_cost, RunTask1);
  }

  void BatchSequentialCompute(
      OpKernelContext* ctx,
      const Eigen::TensorMap<Eigen::Tensor<const string, 1, 1, long>, 16, Eigen::MakePointer> input_vec,
      const int64 batch_size,
      StringPiece sep) {
    std::vector<int64> tokens;
    static constexpr int kReserveSize = 4;
    tokens.reserve(batch_size * kReserveSize);
    int64 output_size = 0;
    int64 max_num_entries = 0;

    std::vector<int64> num_indices(batch_size);
    for (int64 i = 0; i < batch_size; ++i) {
      std::vector<int64> parts = SplitV2(input_vec(i), sep, num_buckets_, maxsplit_);
      int64 n_entries = parts.size();
      num_indices[i] = n_entries;
      output_size += n_entries;
      max_num_entries = std::max(max_num_entries, n_entries);
      tokens.insert(tokens.end(), parts.begin(), parts.end());
    }

    Tensor* sp_indices_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({output_size, 2}), &sp_indices_t));
    Tensor* sp_tokens_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape({output_size}), &sp_tokens_t));
    Tensor* sp_shape_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, TensorShape({2}), &sp_shape_t));

    auto sp_indices = sp_indices_t->matrix<int64>();
    auto sp_tokens = sp_tokens_t->vec<int64>();
    auto sp_shape = sp_shape_t->vec<int64>();
    sp_shape(0) = batch_size;
    sp_shape(1) = max_num_entries;
    size_t c = 0;
    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t j = 0; j < num_indices[i]; ++j) {
        sp_indices(c, 0) = i;
        sp_indices(c, 1) = j;
        sp_tokens(c) = tokens[c];
        ++c;
      }
    }
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input_tensor));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(input_tensor->shape()),
                errors::InvalidArgument("input must be a vector, got shape: ",
                                        input_tensor->shape().DebugString()));

    const auto input_vec = input_tensor->vec<string>();
    const int64 batch_size = input_vec.dimension(0);

    const Tensor* sep_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("sep", &sep_tensor));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(sep_tensor->shape()),
                errors::InvalidArgument("sep must be a scalar, got shape: ",
                                        sep_tensor->shape().DebugString()));
    const auto sep_vec = sep_tensor->flat<string>();
    StringPiece sep(sep_vec(0));

    uint64 start = 0;
    uint64 end = 0;

    if (element_cost_ == 0 && batch_size) {
      size_t sample_id = rand() % batch_size;
      std::vector<int64> temp_for_warm_up =
          SplitV2(input_vec(sample_id), sep, num_buckets_, maxsplit_);
      start = Env::Default()->NowNanos();
      temp_for_warm_up = SplitV2(input_vec(sample_id), sep, num_buckets_, maxsplit_);
      end = Env::Default()->NowNanos();
      element_cost_ = end -start;
    }
    uint64 element_cost = element_cost_;
    if (element_cost * batch_size >= parallel_limit_) {
      BatchParallelCompute(ctx, input_vec, batch_size, sep);
    } else {
      BatchSequentialCompute(ctx, input_vec, batch_size, sep);
    }
  }

 private:
  int maxsplit_;
  uint64 element_cost_;
  uint64 result_cost_;
  const int64 parallel_limit_ = 24;
  int64 num_buckets_;
};

REGISTER_KERNEL_BUILDER(Name("StringSplitToHashBucket") \
                            .Device(DEVICE_CPU),        \
                        StringSplitToHashBucketOp);

}  // namespace tensorflow
