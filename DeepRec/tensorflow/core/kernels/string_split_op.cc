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
#include <string.h>
#include <stdlib.h>
#include <immintrin.h>
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
      const int kReserveSize = 64;
      int numReserve = batch_size * kReserveSize / num_threads ?
                       batch_size * kReserveSize / num_threads : 1;
      tokens_buffer.reserve(numReserve);
      num_indices_buffer.assign(batch_size, 0);
  }

  std::vector<StringPiece> tokens_buffer;
  int64 output_size;
  int64 counter_for_thread;
  int64 thread_index;
  int64 max_num_entries;
  std::vector<int64> num_indices_buffer;
};

template <typename Predicate>
std::vector<StringPiece> SplitOnChar(const string& str,
                                     const char delim, Predicate p) {
  std::vector<StringPiece> result;
  StringPiece text(str);
  auto f = text.find(delim);
  while (f != StringPiece::npos) {
    StringPiece token = text.substr(0, f);
    if (p(token)) {
      result.emplace_back(token);
    }
    text.remove_prefix(f + 1);
    f = text.find(delim);
  }
  if (p(text)) {
    result.push_back(text);
  }
  return result;
}

// Split input string `str` based on a set of character delimiters.
// Returns a vector of StringPieces which are valid as long as input `str`
// is valid.
// Based on str_util::Split.
template <typename Predicate>
std::vector<StringPiece> SplitOnCharSet(const string& str,
                                        const string& delim_set, Predicate p) {
  std::vector<StringPiece> result;
  StringPiece text(str);
  StringPiece delims(delim_set);
  size_t token_start = 0;
  for (size_t i = 0; i < text.size() + 1; i++) {
    if ((i == text.size()) || (delims.find(text[i]) != StringPiece::npos)) {
      StringPiece token(text.data() + token_start, i - token_start);
      if (p(token)) {
        result.emplace_back(token);
      }
      token_start = i + 1;
    }
  }
  return result;
}

// Split input string `str` based on given delimiter.
// Returns a vector of StringPieces which are valid as long as input `str`
// is valid.
template <typename Predicate>
std::vector<StringPiece> Split(const string& str, const string& delimiter,
                               Predicate predicate) {
  if (str.empty()) {
    return std::vector<StringPiece>();
  }
  if (delimiter.empty()) {
    std::vector<StringPiece> result;
    result.resize(str.size());
    for (size_t i = 0; i < str.size(); ++i) {
      result[i] = StringPiece(str.data() + i, 1);
    }
    return result;
  }
  if (delimiter.size() == 1) {
    return SplitOnChar(str, delimiter[0], predicate);
  }
  return SplitOnCharSet(str, delimiter, predicate);
}

std::vector<StringPiece> SplitV2(const string& str, StringPiece sep,
                                 int maxsplit) {
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

  std::vector<StringPiece> result;

  StringPiece text(str);
  if (maxsplit == 0) {
    result.emplace_back(text);
    return result;
  }

  if (sep.empty()) {
    StringPiece token;
    // Remove leading whitespaces.
    str_util::RemoveLeadingWhitespace(&text);
    int split = 0;
    while (str_util::ConsumeNonWhitespace(&text, &token)) {
      result.push_back(token);
      str_util::RemoveLeadingWhitespace(&text);
      ++split;
      if (maxsplit > 0 && split == maxsplit) {
        result.push_back(text);
        return result;
      }
    }
    return result;
  }
  auto p = std::search(text.begin(), text.end(), sep.begin(), sep.end());
  int split = 0;
  while (p != text.end()) {
    StringPiece token = text.substr(0, p - text.begin());
    result.push_back(token);
    text.remove_prefix(token.size());
    text.remove_prefix(sep.size());
    ++split;
    if (maxsplit > 0 && split == maxsplit) {
      result.push_back(StringPiece(text));
      return result;
    }
    p = std::search(text.begin(), text.end(), sep.begin(), sep.end());
  }
  result.push_back(text);
  return result;
}

}  // namespace


class StringSplitOp : public OpKernel {
 public:
  explicit StringSplitOp(OpKernelConstruction* context)
      : OpKernel(context), skip_empty_(true),
      element_cost_(0), result_cost_(0) {
    bool skip_empty;
    // By default skip_empty_ is true. We only get the value from attr if it is
    // available, so that it is backward compatible.
    if (context->GetAttr("skip_empty", &skip_empty).ok()) {
      skip_empty_ = skip_empty;
    }
  }

  void ParallelSplit(OpKernelContext* ctx,
                      const Eigen::TensorMap<
                        Eigen::Tensor<const string, 1, 1, long>,
                        16, Eigen::MakePointer> input_vec,
                      const int64 batch_size, const string& delimiter) {
    ThreadPool* thread_pool =
    ctx->device()->tensorflow_cpu_worker_threads()->workers;
    const int64 num_threads = thread_pool->NumThreads() + 1;

    std::vector<int64> num_indices(batch_size);
    num_indices[0] = 0;

    std::vector<WorkerInfo> w_array;
    for (int i = 0; i < num_threads; i++) {
      WorkerInfo w(num_threads, batch_size);
      w_array.emplace_back(w);
    }
    std::vector<std::vector<int64>> id_to_worker(batch_size);

    thread_pool->ParallelForWithWorkerId(
      batch_size,
      element_cost_,
      [&w_array, &id_to_worker, &input_vec,
      &delimiter, ctx, this, &num_indices]
      (int64 start, int64 end, int64 worker_id){
        int64 position_in_worker = 0;
        for (int64 i = start; i < end; ++i) {
          std::vector<StringPiece> parts =
              skip_empty_ ?
               Split(input_vec(i), delimiter, str_util::SkipEmpty())
               : Split(input_vec(i), delimiter, str_util::AllowEmpty());
          int64 n_entries = parts.size();
          id_to_worker[i].emplace_back(worker_id);
          id_to_worker[i].emplace_back(w_array[worker_id].counter_for_thread);
          id_to_worker[i].emplace_back(w_array[worker_id].output_size);
          num_indices[i] = n_entries;
          position_in_worker += n_entries;
          w_array[worker_id].num_indices_buffer[
              w_array[worker_id].counter_for_thread] = n_entries;
          w_array[worker_id].output_size += n_entries;
          w_array[worker_id].max_num_entries =
              std::max(w_array[worker_id].max_num_entries, n_entries);
          w_array[worker_id].tokens_buffer.insert(
              w_array[worker_id].tokens_buffer.end(),
              std::make_move_iterator(parts.begin()),
              std::make_move_iterator(parts.end()));
          w_array[worker_id].counter_for_thread++;
       }
     });

    int64 output_size = 0;
    int64 max_num_entries = 0;
    for (int i = 0; i < num_threads; i++) {
      output_size += w_array[i].output_size;
      max_num_entries = std::max(w_array[i].max_num_entries, max_num_entries);
    }

    std::vector<int64> id_to_index(batch_size);
    id_to_index[0] = 0;
    for (size_t i = 1; i < batch_size; i++)
      id_to_index[i] = id_to_index[i-1] + num_indices[i-1];

    Tensor* sp_indices_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({output_size, 2}),
                                             &sp_indices_t));
    Tensor* sp_tokens_t;
    OP_REQUIRES_OK(
       ctx, ctx->allocate_output(1, TensorShape({output_size}), &sp_tokens_t));
    Tensor* sp_shape_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, TensorShape({2}), &sp_shape_t));

    auto sp_indices = sp_indices_t->matrix<int64>();
    auto sp_tokens = sp_tokens_t->vec<string>();
    auto sp_shape = sp_shape_t->vec<int64>();
    sp_shape(0) = batch_size;
    sp_shape(1) = max_num_entries;

    if (result_cost_ == 0) {
      uint64 start = 0;
      uint64 end = 0;
      size_t sample_id = rand() % batch_size;
      start = Env::Default()->NowNanos();
      int last_worker_id = id_to_worker[sample_id][0];
      int64 id_in_last_worker = id_to_worker[sample_id][1];
      int64 st = id_to_worker[sample_id][2];
      for (int64 j = 0;
          j < w_array[last_worker_id].num_indices_buffer[id_in_last_worker];
          j++) {
        size_t c = id_to_index[sample_id] + j;
        sp_indices(c, 0) = sample_id;
        sp_indices(c, 1) = j;
        sp_tokens(c).assign(
             w_array[last_worker_id].tokens_buffer[st+j].data(),
             w_array[last_worker_id].tokens_buffer[st+j].size());
      }
      end = Env::Default()->NowNanos();
      result_cost_ = end - start;
    }
    uint64 result_cost = result_cost_;

    thread_pool->ParallelForWithWorkerId(
      batch_size,
      result_cost,
      [&id_to_index, batch_size, &w_array,
       &id_to_worker, &sp_indices,
       &num_indices, &sp_tokens](int64 start, int64 end, int64 worker_id) {
        for (int64 i = start; i < end; i++) {
          int last_worker_id = id_to_worker[i][0];
          int64 id_in_last_worker = id_to_worker[i][1];
          int64 st = id_to_worker[i][2];
          for (int64 j = 0;
              j < w_array[last_worker_id].num_indices_buffer[id_in_last_worker];
              j++) {
              size_t c = id_to_index[i] + j;
              sp_indices(c, 0) = i;
              sp_indices(c, 1) = j;
              sp_tokens(c).assign(
                  w_array[last_worker_id].tokens_buffer[st+j].data(),
                  w_array[last_worker_id].tokens_buffer[st+j].size());
          }
        }
      });
  }

  void SequentialSplit(OpKernelContext* ctx,
                        const Eigen::TensorMap<
                        Eigen::Tensor<const string, 1, 1, long>,
                        16, Eigen::MakePointer> input_vec,
                        const int64 batch_size, const string& delimiter) {
    std::vector<StringPiece> tokens;
    static constexpr int kReserveSize = 4;
    tokens.reserve(batch_size * kReserveSize);
    int64 output_size = 0;
    int64 max_num_entries = 0;

    std::vector<int64> num_indices(batch_size);
    for (int64 i = 0; i < batch_size; ++i) {
      std::vector<StringPiece> parts =
          skip_empty_ ? Split(input_vec(i), delimiter, str_util::SkipEmpty())
                      : Split(input_vec(i), delimiter, str_util::AllowEmpty());
      int64 n_entries = parts.size();
      num_indices[i] = n_entries;
      output_size += n_entries;
      max_num_entries = std::max(max_num_entries, n_entries);
      tokens.insert(tokens.end(), std::make_move_iterator(parts.begin()),
                    std::make_move_iterator(parts.end()));
    }

    Tensor* sp_indices_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({output_size, 2}),
                                             &sp_indices_t));
    Tensor* sp_tokens_t;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(1, TensorShape({output_size}), &sp_tokens_t));
    Tensor* sp_shape_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, TensorShape({2}), &sp_shape_t));

    auto sp_indices = sp_indices_t->matrix<int64>();
    auto sp_tokens = sp_tokens_t->vec<string>();
    auto sp_shape = sp_shape_t->vec<int64>();
    sp_shape(0) = batch_size;
    sp_shape(1) = max_num_entries;
    size_t c = 0;
    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t j = 0; j < num_indices[i]; ++j) {
        sp_indices(c, 0) = i;
        sp_indices(c, 1) = j;
        sp_tokens(c).assign(tokens[c].data(), tokens[c].size());
        ++c;
      }
    }
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* input_tensor;
    srand((unsigned)time(NULL));

    OP_REQUIRES_OK(ctx, ctx->input("input", &input_tensor));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(input_tensor->shape()),
                errors::InvalidArgument("input must be a vector, got shape: ",
                                        input_tensor->shape().DebugString()));

    const auto input_vec = input_tensor->vec<string>();
    const int64 batch_size = input_vec.dimension(0);

    const Tensor* delimiter_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("delimiter", &delimiter_tensor));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(delimiter_tensor->shape()),
        errors::InvalidArgument("delimiter must be a scalar, got shape: ",
                                delimiter_tensor->shape().DebugString()));
    const auto delimiter_vec = delimiter_tensor->flat<string>();
    const string& delimiter = delimiter_vec(0);

    uint64 start = 0;
    uint64 end = 0;

    if (element_cost_ == 0 && batch_size) {
      size_t sample_id = rand() % batch_size;
      std::vector<StringPiece> temp_for_warm_up =
            skip_empty_ ?
               Split(input_vec(sample_id), delimiter, str_util::SkipEmpty())
               : Split(input_vec(sample_id), delimiter, str_util::AllowEmpty());
      start = Env::Default()->NowNanos();
      temp_for_warm_up =
            skip_empty_ ?
             Split(input_vec(sample_id), delimiter, str_util::SkipEmpty())
             : Split(input_vec(sample_id), delimiter, str_util::AllowEmpty());
      end = Env::Default()->NowNanos();
      element_cost_ = end - start;
    }
    uint64 element_cost = element_cost_;

    if (element_cost * batch_size >= parallel_limit_) {
      ParallelSplit(ctx, input_vec, batch_size, delimiter);
    } else {
      SequentialSplit(ctx, input_vec, batch_size, delimiter);
    }
  }

 private:
  bool skip_empty_;
  uint64 element_cost_;
  uint64 result_cost_;
  const int64 parallel_limit_ = 240000;
};

class StringSplitV2Op : public OpKernel {
 public:
  explicit StringSplitV2Op(OpKernelConstruction* context)
      : OpKernel(context), maxsplit_(-1), element_cost_(0), result_cost_(0) {
    OP_REQUIRES_OK(context, context->GetAttr("maxsplit", &maxsplit_));
  }

  void ParallelSplitV2(OpKernelContext* ctx,
                      const Eigen::TensorMap<
                      Eigen::Tensor<const string, 1, 1, long>,
                      16, Eigen::MakePointer> input_vec,
                      const int64 batch_size, StringPiece sep) {
    ThreadPool* thread_pool =
        ctx->device()->tensorflow_cpu_worker_threads()->workers;
    const int64 num_threads = thread_pool->NumThreads() + 1;

    std::vector<int64> num_indices(batch_size);
    num_indices[0] = 0;

    std::vector<WorkerInfo> w_array;
    for (int i = 0; i < num_threads; i++) {
      WorkerInfo w(num_threads, batch_size);
      w_array.emplace_back(w);
    }

    std::vector<std::vector<int64>> id_to_worker(batch_size);
    thread_pool->ParallelForWithWorkerId(
      batch_size,
      element_cost_,
      [&sep, &w_array, &id_to_worker,
       &input_vec, ctx, this,
       &num_indices](int64 start, int64 end, int64 worker_id) {
        int64 position_in_worker = 0;
        for (int64 i = start; i < end; ++i) {
          std::vector<StringPiece> parts =
              SplitV2(input_vec(i), sep, maxsplit_);
          int64 n_entries = parts.size();
          id_to_worker[i].emplace_back(worker_id);
          id_to_worker[i].emplace_back(w_array[worker_id].counter_for_thread);
          id_to_worker[i].emplace_back(w_array[worker_id].output_size);
          num_indices[i] = n_entries;
          position_in_worker += n_entries;
          w_array[worker_id].num_indices_buffer[
              w_array[worker_id].counter_for_thread] = n_entries;
          w_array[worker_id].output_size += n_entries;
          w_array[worker_id].max_num_entries =
              std::max(w_array[worker_id].max_num_entries, n_entries);
          w_array[worker_id].tokens_buffer.insert(
               w_array[worker_id].tokens_buffer.end(),
               std::make_move_iterator(parts.begin()),
               std::make_move_iterator(parts.end()));
          w_array[worker_id].counter_for_thread++;
       }
      });

    int64 output_size = 0;
    int64 max_num_entries = 0;
    for (int i = 0; i < num_threads; i++) {
      output_size += w_array[i].output_size;
      max_num_entries = std::max(w_array[i].max_num_entries, max_num_entries);
    }


    std::vector<int64> id_to_index(batch_size);
    id_to_index[0] = 0;
    for (size_t i = 1; i < batch_size; i++)
      id_to_index[i] = id_to_index[i-1] + num_indices[i-1];


    Tensor* sp_indices_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({output_size, 2}),
                                             &sp_indices_t));
    Tensor* sp_tokens_t;
    OP_REQUIRES_OK(
       ctx, ctx->allocate_output(1, TensorShape({output_size}), &sp_tokens_t));
    Tensor* sp_shape_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, TensorShape({2}), &sp_shape_t));

    auto sp_indices = sp_indices_t->matrix<int64>();
    auto sp_tokens = sp_tokens_t->vec<string>();
    auto sp_shape = sp_shape_t->vec<int64>();
    sp_shape(0) = batch_size;
    sp_shape(1) = max_num_entries;

    if (result_cost_ == 0) {
      uint64 start = 0;
      uint64 end = 0;
      size_t sample_id = rand() % batch_size;
      start = Env::Default()->NowNanos();
      int last_worker_id = id_to_worker[sample_id][0];
      int64 id_in_last_worker = id_to_worker[sample_id][1];
      int64 st = id_to_worker[sample_id][2];
      for (int64 j = 0;
       j < w_array[last_worker_id].num_indices_buffer[id_in_last_worker];
       j++) {
        size_t c = id_to_index[sample_id] + j;
        sp_indices(c, 0) = sample_id;
        sp_indices(c, 1) = j;
        sp_tokens(c).assign(
            w_array[last_worker_id].tokens_buffer[st+j].data(),
            w_array[last_worker_id].tokens_buffer[st+j].size());
      }
      end = Env::Default()->NowNanos();
      result_cost_ = end - start;
    }
    uint64 result_cost = result_cost_;

    thread_pool->ParallelForWithWorkerId(
      batch_size,
      result_cost,
      [&id_to_index, batch_size, &w_array,
       &id_to_worker, &sp_indices, &num_indices,
       &sp_tokens](int64 start, int64 end, int64 worker_id) {
        for (int64 i = start; i < end; i++) {
          int last_worker_id = id_to_worker[i][0];
          int64 id_in_last_worker = id_to_worker[i][1];
          int64 st = id_to_worker[i][2];
          for (int64 j = 0;
           j < w_array[last_worker_id].num_indices_buffer[id_in_last_worker];
           j++) {
              size_t c = id_to_index[i] + j;
              sp_indices(c, 0) = i;
              sp_indices(c, 1) = j;
              sp_tokens(c).assign(
                  w_array[last_worker_id].tokens_buffer[st+j].data(),
                  w_array[last_worker_id].tokens_buffer[st+j].size());
          }
        }
      });
  }

  void SequentialSplitV2(OpKernelContext* ctx,
                          const Eigen::TensorMap<
                          Eigen::Tensor<const string, 1, 1, long>,
                          16, Eigen::MakePointer> input_vec,
                          const int64 batch_size, StringPiece sep) {
    std::vector<StringPiece> tokens;
    static constexpr int kReserveSize = 4;
    tokens.reserve(batch_size * kReserveSize);
    int64 output_size = 0;
    int64 max_num_entries = 0;

    std::vector<int64> num_indices(batch_size);
    for (int64 i = 0; i < batch_size; ++i) {
      std::vector<StringPiece> parts = SplitV2(input_vec(i), sep, maxsplit_);
      int64 n_entries = parts.size();
      num_indices[i] = n_entries;
      output_size += n_entries;
      max_num_entries = std::max(max_num_entries, n_entries);
      tokens.insert(tokens.end(), parts.begin(), parts.end());
    }

    Tensor* sp_indices_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({output_size, 2}),
                                             &sp_indices_t));
    Tensor* sp_tokens_t;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(1, TensorShape({output_size}), &sp_tokens_t));
    Tensor* sp_shape_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, TensorShape({2}), &sp_shape_t));

    auto sp_indices = sp_indices_t->matrix<int64>();
    auto sp_tokens = sp_tokens_t->vec<string>();
    auto sp_shape = sp_shape_t->vec<int64>();
    sp_shape(0) = batch_size;
    sp_shape(1) = max_num_entries;
    size_t c = 0;
    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t j = 0; j < num_indices[i]; ++j) {
        sp_indices(c, 0) = i;
        sp_indices(c, 1) = j;
        sp_tokens(c).assign(tokens[c].data(), tokens[c].size());
        ++c;
      }
    }
  }

#if defined(__AVX512F__)
  void MklSplitChar(
      OpKernelContext* ctx,
      const Eigen::TensorMap<Eigen::Tensor<const string, 1, 1, long>, 16,
                             Eigen::MakePointer>
          input_vec,
      const int64 batch_size, const char sep, const bool try_parallel) {
    ThreadPool* thread_pool =
        ctx->device()->tensorflow_cpu_worker_threads()->workers;

    std::vector<int64> output_size_arr(batch_size);
    output_size_arr[0] = 0;
    __m512i sep512 = _mm512_set1_epi8(sep);
    auto RunTaskCnt = [this, &input_vec, &sep512, &output_size_arr](
                          int64 start, int64 end, int64 worker_id) {
      for (int64 batch_idx = start; batch_idx < end; batch_idx++) {
        int64 seq_cnt = 0;  // count of sep for one string
        int64 seq_idx = 0;
        int64 text_size = (int64)input_vec(batch_idx).size();
        __m512i x;
        __mmask64 mask;
        for (; seq_idx + 63 < text_size; seq_idx += 64) {
          x = _mm512_loadu_si512(input_vec(batch_idx).data() + seq_idx);
          mask = _mm512_cmpeq_epi8_mask(x, sep512);
          seq_cnt += _mm_popcnt_u64(_cvtmask64_u64(mask));
        }
        if (seq_idx != text_size) {
          x = _mm512_mask_loadu_epi8(
              _mm512_setzero_si512(),
              0xFFFFFFFFFFFFFFFF >> (seq_idx - text_size),
              input_vec(batch_idx).data() + seq_idx);
          mask = _mm512_cmpeq_epi8_mask(x, sep512);
          seq_cnt += _mm_popcnt_u64(_cvtmask64_u64(mask));
        }
        output_size_arr[batch_idx] = (++seq_cnt);
      }
    };

    uint64 t0, t1, t3, task1_element_cost = 0;
    if (try_parallel) {
      t0 = Env::Default()->NowNanos();
      RunTaskCnt(0, 1, 0);
      t1 = Env::Default()->NowNanos();
      task1_element_cost = t1 - t0;
    }
    if (try_parallel && task1_element_cost * batch_size > parallel_limit_) {
      thread_pool->ParallelForWithWorkerId(batch_size, task1_element_cost,
                                           RunTaskCnt);
    } else {
      RunTaskCnt(0, batch_size, 0);
    }
    t3 = Env::Default()->NowNanos();

    int64 max_num_entries = output_size_arr[0];
    std::vector<int64> idx_start_arr(batch_size);
    idx_start_arr[0] = 0;
    for (int i = 1; i < batch_size; i++) {
      idx_start_arr[i] = idx_start_arr[i - 1] + output_size_arr[i - 1];
      max_num_entries = std::max(max_num_entries, output_size_arr[i]);
    }
    int64 output_size = 0;
    if (batch_size > 0)
      output_size =
          idx_start_arr[batch_size - 1] + output_size_arr[batch_size - 1];

    Tensor* sp_indices_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({output_size, 2}),
                                             &sp_indices_t));
    Tensor* sp_tokens_t;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(1, TensorShape({output_size}), &sp_tokens_t));
    Tensor* sp_shape_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, TensorShape({2}), &sp_shape_t));
    auto sp_indices = sp_indices_t->matrix<int64>();
    auto sp_tokens = sp_tokens_t->vec<string>();
    auto sp_shape = sp_shape_t->vec<int64>();
    sp_shape(0) = batch_size;
    sp_shape(1) = max_num_entries;

    auto RunTask = [this, &input_vec, &sp_indices, &sp_tokens, &sep512,
                    &idx_start_arr](int64 start, int64 end, int64 worker_id) {
      for (int64 batch_idx = start; batch_idx < end; batch_idx++) {
        int64 total_c = idx_start_arr[batch_idx];
        int64 remain_size = 0;
        int64 offset = 0;
        int64 seq_c = 0;  // count of sep for one string
        __m512i x;
        __mmask64 mask = 0, temp_mask = 0;
        int64 seq_idx = 0;
        int64 text_size = (int64)input_vec(batch_idx).size();
        for (; seq_idx + 63 < text_size; seq_idx += 64) {
          x = _mm512_loadu_si512(input_vec(batch_idx).data() + seq_idx);
          mask = _mm512_cmpeq_epi8_mask(x, sep512);
          temp_mask = mask;
          int64 part_cnt = _mm_popcnt_u64(_cvtmask64_u64(mask));
          for (int64 k = 0; k < part_cnt; k++) {
            int64 lz = (int64)_tzcnt_u64(_cvtmask64_u64(temp_mask));
            int64 len = (k == 0 ? lz + remain_size : lz);
            sp_indices(total_c, 0) = batch_idx;
            sp_indices(total_c, 1) = seq_c;
            sp_tokens(total_c).assign(input_vec(batch_idx).data() + offset,
                                      len);
            seq_c++;
            total_c++;
            temp_mask = temp_mask >> (lz + 1);
            offset += (len + 1);
          }
          remain_size = (int64)_lzcnt_u64(_cvtmask64_u64(mask));
        }
        if (seq_idx != text_size) {
          x = _mm512_mask_loadu_epi8(
              _mm512_setzero_si512(),
              0xFFFFFFFFFFFFFFFF >> (seq_idx - text_size),
              input_vec(batch_idx).data() + seq_idx);
          mask = _mm512_cmpeq_epi8_mask(x, sep512);
          temp_mask = mask;
          int64 part_cnt = _mm_popcnt_u64(_cvtmask64_u64(mask));
          for (int64 k = 0; k < part_cnt; k++) {
            int64 lz = (int64)_tzcnt_u64(_cvtmask64_u64(temp_mask));
            int64 len = (k == 0 ? lz + remain_size : lz);
            sp_indices(total_c, 0) = batch_idx;
            sp_indices(total_c, 1) = seq_c;
            sp_tokens(total_c).assign(input_vec(batch_idx).data() + offset,
                                      len);
            seq_c++;
            total_c++;
            temp_mask = temp_mask >> (lz + 1);
            offset += (len + 1);
          }
        }
        sp_indices(total_c, 0) = batch_idx;
        sp_indices(total_c, 1) = seq_c;
        sp_tokens(total_c).assign(input_vec(batch_idx).data() + offset,
                                  text_size - offset);
        seq_c++;
        total_c++;
      }
    };

    uint64 t4, t5, t7, task2_element_cost = 0;
    if (try_parallel) {
      t4 = Env::Default()->NowNanos();
      RunTask(0, 1, 0);
      t5 = Env::Default()->NowNanos();
      task2_element_cost = t5 - t4;
    }
    if (try_parallel && task2_element_cost * batch_size > parallel_limit_) {
      thread_pool->ParallelForWithWorkerId(batch_size, task2_element_cost,
                                           RunTask);
    } else {
      RunTask(0, batch_size, 0);
    }
    t7 = Env::Default()->NowNanos();
  }

  void MklSplitCharWithMaxSplit(
      OpKernelContext* ctx,
      const Eigen::TensorMap<Eigen::Tensor<const string, 1, 1, long>, 16,
                             Eigen::MakePointer>
          input_vec,
      const int64 batch_size, const char sep, const bool try_parallel) {
    ThreadPool* thread_pool =
        ctx->device()->tensorflow_cpu_worker_threads()->workers;

    std::vector<int64> output_size_arr(batch_size);
    output_size_arr[0] = 0;
    __m512i sep512 = _mm512_set1_epi8(sep);
    auto RunTaskCnt = [this, &input_vec, &sep512, &output_size_arr](
                          int64 start, int64 end, int64 worker_id) {
      for (int64 batch_idx = start; batch_idx < end; batch_idx++) {
        int64 seq_cnt = 0;  // count of sep for one string
        int64 seq_idx = 0;
        int64 text_size = (int64)input_vec(batch_idx).size();
        __m512i x;
        __mmask64 mask;
        for (; seq_idx + 63 < text_size; seq_idx += 64) {
          x = _mm512_loadu_si512(input_vec(batch_idx).data() + seq_idx);
          mask = _mm512_cmpeq_epi8_mask(x, sep512);
          seq_cnt += _mm_popcnt_u64(_cvtmask64_u64(mask));
          if (maxsplit_ > 0 && seq_cnt >= maxsplit_) break;
        }
        if (seq_idx != text_size) {
          x = _mm512_mask_loadu_epi8(
              _mm512_setzero_si512(),
              0xFFFFFFFFFFFFFFFF >> (seq_idx - text_size),
              input_vec(batch_idx).data() + seq_idx);
          mask = _mm512_cmpeq_epi8_mask(x, sep512);
          seq_cnt += _mm_popcnt_u64(_cvtmask64_u64(mask));
        }
        ++seq_cnt;
        output_size_arr[batch_idx] =
            (maxsplit_ > 0 && seq_cnt >= maxsplit_) ? maxsplit_ : seq_cnt;
      }
    };

    uint64 t0, t1, t3, task1_element_cost = 0;
    if (try_parallel) {
      t0 = Env::Default()->NowNanos();
      RunTaskCnt(0, 1, 0);
      t1 = Env::Default()->NowNanos();
      task1_element_cost = t1 - t0;
    }
    if (task1_element_cost * batch_size > parallel_limit_) {
      thread_pool->ParallelForWithWorkerId(batch_size, task1_element_cost,
                                           RunTaskCnt);
    } else {
      RunTaskCnt(0, batch_size, 0);
    }
    t3 = Env::Default()->NowNanos();

    int64 max_num_entries = output_size_arr[0];
    std::vector<int64> idx_start_arr(batch_size);
    idx_start_arr[0] = 0;
    for (int i = 1; i < batch_size; i++) {
      idx_start_arr[i] = idx_start_arr[i - 1] + output_size_arr[i - 1];
      max_num_entries = std::max(max_num_entries, output_size_arr[i]);
    }
    int64 output_size = 0;
    if (batch_size > 0)
      output_size =
          idx_start_arr[batch_size - 1] + output_size_arr[batch_size - 1];

    Tensor* sp_indices_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({output_size, 2}),
                                             &sp_indices_t));
    Tensor* sp_tokens_t;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(1, TensorShape({output_size}), &sp_tokens_t));
    Tensor* sp_shape_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, TensorShape({2}), &sp_shape_t));
    auto sp_indices = sp_indices_t->matrix<int64>();
    auto sp_tokens = sp_tokens_t->vec<string>();
    auto sp_shape = sp_shape_t->vec<int64>();
    sp_shape(0) = batch_size;
    sp_shape(1) = max_num_entries;

    auto RunTask = [this, &input_vec, &sp_indices, &sp_tokens, &sep512,
                    &idx_start_arr](int64 start, int64 end, int64 worker_id) {
      for (int64 batch_idx = start; batch_idx < end; batch_idx++) {
        int64 total_c = idx_start_arr[batch_idx];
        int64 remain_size = 0;
        int64 offset = 0;
        int64 seq_c = 0;  // count of sep for one string
        __m512i x;
        __mmask64 mask = 0, temp_mask = 0;
        int64 seq_idx = 0;
        int64 text_size = (int64)input_vec(batch_idx).size();
        bool overflow = false;
        for (; seq_idx + 63 < text_size; seq_idx += 64) {
          x = _mm512_loadu_si512(input_vec(batch_idx).data() + seq_idx);
          mask = _mm512_cmpeq_epi8_mask(x, sep512);
          temp_mask = mask;
          int64 part_cnt = _mm_popcnt_u64(_cvtmask64_u64(mask));
          if (maxsplit_ > 0 && (part_cnt + seq_c) >= maxsplit_) {
            overflow = true;
            part_cnt = maxsplit_ - seq_c;
          }
          for (int64 k = 0; k < part_cnt; k++) {
            int64 lz = (int64)_tzcnt_u64(_cvtmask64_u64(temp_mask));
            int64 len = (k == 0 ? lz + remain_size : lz);
            sp_indices(total_c, 0) = batch_idx;
            sp_indices(total_c, 1) = seq_c;
            sp_tokens(total_c).assign(input_vec(batch_idx).data() + offset,
                                      len);
            seq_c++;
            total_c++;
            temp_mask = temp_mask >> (lz + 1);
            offset += (len + 1);
          }
          if (overflow) break;
          remain_size = (int64)_lzcnt_u64(_cvtmask64_u64(mask));
        }
        if (seq_idx != text_size && overflow != true) {
          x = _mm512_mask_loadu_epi8(
              _mm512_setzero_si512(),
              0xFFFFFFFFFFFFFFFF >> (seq_idx - text_size),
              input_vec(batch_idx).data() + seq_idx);
          mask = _mm512_cmpeq_epi8_mask(x, sep512);
          temp_mask = mask;
          int64 part_cnt = _mm_popcnt_u64(_cvtmask64_u64(mask));
          if (maxsplit_ > 0 && (part_cnt + seq_c) >= maxsplit_) {
            part_cnt = maxsplit_ - seq_c;
            overflow = true;
          }
          for (int64 k = 0; k < part_cnt; k++) {
            int64 lz = (int64)_tzcnt_u64(_cvtmask64_u64(temp_mask));
            int64 len = (k == 0 ? lz + remain_size : lz);
            sp_indices(total_c, 0) = batch_idx;
            sp_indices(total_c, 1) = seq_c;
            sp_tokens(total_c).assign(input_vec(batch_idx).data() + offset,
                                      len);
            seq_c++;
            total_c++;
            temp_mask = temp_mask >> (lz + 1);
            offset += (len + 1);
          }
        }
        if (!overflow) {
          sp_indices(total_c, 0) = batch_idx;
          sp_indices(total_c, 1) = seq_c;
          sp_tokens(total_c).assign(input_vec(batch_idx).data() + offset,
                                    text_size - offset);
          seq_c++;
          total_c++;
        }
      }
    };

    uint64 t4, t5, t7, task2_element_cost = 0;
    if (try_parallel) {
      t4 = Env::Default()->NowNanos();
      RunTask(0, 1, 0);
      t5 = Env::Default()->NowNanos();
      task2_element_cost = t5 - t4;
    }
    if (task2_element_cost * batch_size > parallel_limit_) {
      thread_pool->ParallelForWithWorkerId(batch_size, task2_element_cost,
                                           RunTask);
    } else {
      RunTask(0, batch_size, 0);
    }
    t7 = Env::Default()->NowNanos();
  }
#endif

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

    if (element_cost_ == 0 && batch_size &&
        (sep_vec(0).size() != 1 || maxsplit_ > 0)) {
      size_t sample_id = rand() % batch_size;
      std::vector<StringPiece> temp_for_warm_up =
          SplitV2(input_vec(sample_id), sep, maxsplit_);
      start = Env::Default()->NowNanos();
      temp_for_warm_up = SplitV2(input_vec(sample_id), sep, maxsplit_);
      end = Env::Default()->NowNanos();
      element_cost_ = end - start;
    }
    uint64 element_cost = element_cost_;
#if defined(__AVX512F__)
    if (sep_vec(0).size() == 1 && batch_size > 0) {
      char sep = *(sep_vec(0).data());
      bool try_parallel = input_tensor->TotalBytes() > 819200;
      if (maxsplit_ <= 0)
        MklSplitChar(ctx, input_vec, batch_size, sep, try_parallel);
      else
        MklSplitCharWithMaxSplit(ctx, input_vec, batch_size, sep, try_parallel);
    } else if (element_cost * batch_size >= parallel_limit_) {
      ParallelSplitV2(ctx, input_vec, batch_size, sep);
    } else {
      SequentialSplitV2(ctx, input_vec, batch_size, sep);
    }
#else
    if (element_cost * batch_size >= parallel_limit_) {
      ParallelSplitV2(ctx, input_vec, batch_size, sep);
    } else {
      SequentialSplitV2(ctx, input_vec, batch_size, sep);
    }
#endif
  }

 private:
  int maxsplit_;
  uint64 element_cost_;
  uint64 result_cost_;
  const int64 parallel_limit_ = 240000;
};

REGISTER_KERNEL_BUILDER(Name("StringSplit").Device(DEVICE_CPU), StringSplitOp);
REGISTER_KERNEL_BUILDER(Name("StringSplitV2").Device(DEVICE_CPU),
                        StringSplitV2Op);
REGISTER_KERNEL_BUILDER(Name("StringSplitSlice").Device(DEVICE_CPU),
                        StringSplitV2Op);

}  // namespace tensorflow
