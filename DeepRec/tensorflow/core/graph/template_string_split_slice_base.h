/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_GRAPH_TEMPLATE_STRING_SPLIT_SLICE_BASE_H_
#define TENSORFLOW_CORE_GRAPH_TEMPLATE_STRING_SPLIT_SLICE_BASE_H_

#include "tensorflow/core/graph/template_base.h"

namespace tensorflow {

class TemplateStringSplitSliceBase : public TemplateBase {
 public:
  TemplateStringSplitSliceBase() {
    const TempNode n01 = {
      .key = "input_layer/StringSplit/StringSplitV2",
      .op = "StringSplitV2",
      .inputs = {/* prefetch_2/Identity_2 */ "0", 
                 /* input_layer/StringSplit/Const */ "1" },
      .outputs = {{"input_layer/SparseSlice"},
                  {"input_layer/SparseSlice"},
                  {"input_layer/SparseSlice"}}};
    temp_nodes_.emplace_back(n01);

    const TempNode n03 = {
      .key = "input_layer/Cast",
      .op = "Cast",
      .inputs = {/* strided_slice */ "4"},
      .outputs = {{"input_layer/packed"}}};
    temp_nodes_.emplace_back(n03);

    const TempNode n02 = {
      .key = "input_layer/packed",
      .op = "Pack",
      .inputs = {"input_layer/Cast",
                 /* input_layer/packed/1 */ "3"},
      .outputs = {{"input_layer/SparseSlice"}}};
    temp_nodes_.emplace_back(n02);

    const TempNode n11 = {
      .key = "input_layer/SparseSlice",
      .op = "SparseSlice",
      .inputs = {"input_layer/StringSplit/StringSplitV2",
                 "input_layer/StringSplit/StringSplitV2", //:2
                 "input_layer/StringSplit/StringSplitV2", //:1
                 /* input_layer/Const */ "2",
                 "input_layer/packed"},
      .outputs = {{/* input_layer/SparseReshape */ "0"},
                  {/* input_layer/hash_table_Lookup_2/LookupTableFindV2 */ "1"},
                  {/* input_layer/SparseReshape */ "2", /* input_layer/Shape/Cast */ "3"}}};
    temp_nodes_.emplace_back(n11);

    first_key_ = "input_layer/StringSplit/StringSplitV2";
    num_inputs_ = 5;
    num_outputs_ = 4;
  }

  const string name() { return "string_split_slice_base"; }

  bool add_subgraph(std::map<std::string, MatchedNode>& nodes,
                    std::string name_prefix, Graph* g,
                    std::vector<const Edge*>& inputs,
                    std::vector<std::vector<const Edge*>>& outputs) override {
    LOG(INFO) << "Fusion template[" << name() << "] match op["
              << nodes[first_key_].node->name() << "][new_name:" << name_prefix
              << "_" << name() << "]";

    Node* node_fused_string_split_slice = add_fused_string_split_slice_node(
        nodes, name_prefix, g, inputs, outputs);
    if (!node_fused_string_split_slice) {
      LOG(WARNING) << "Add node_fused_string_split_slice node failed";
      return false;
    }

    return rebuild_graph(g, inputs, outputs, node_fused_string_split_slice);
  }

  bool CheckDynamicInputs(
      const Node* node, const TempNode* temp_node, int dy_mode,
      std::vector<const Edge*>& fused_op_inputs,
      std::map<const std::string, TempNode>& temp_node_map,
      std::map<std::string, MatchedNode>& matched_node_map) override {
    return false;
  }

  bool CheckDynamicOutputs(
      const Node* node, const TempNode* temp_node, int dy_mode,
      std::vector<std::vector<const Edge*>>& fused_op_outputs,
      std::map<const std::string, TempNode>& temp_node_map,
      std::map<std::string, MatchedNode>& matched_node_map) override {
    return false;
  }

 protected:
  virtual Node* add_fused_string_split_slice_node(
      std::map<std::string, MatchedNode>& nodes, std::string name_prefix,
      Graph* g, std::vector<const Edge*>& inputs,
      std::vector<std::vector<const Edge*>>& outputs) {
    // construct fused_string_split_slice node
    NodeDef fused_string_split_slice_node;
    add_input(fused_string_split_slice_node, inputs[0]);
    add_input(fused_string_split_slice_node, inputs[1]);
    fused_string_split_slice_node.set_op("StringSplitSlice");
    fused_string_split_slice_node.set_name(name_prefix + name());
    fused_string_split_slice_node.set_device(
        nodes["input_layer/StringSplit/StringSplitV2"].node->def().device());
    AttrValue maxsplit_attr;
    Tensor t;
    Status s= GetNodeAttr(((Node*)(inputs[3]->src()))->attrs(), "value", &t);
    // LOG(INFO) << "GetNodeAttr:" << (int64)t.scalar<int64>()(0);
    maxsplit_attr.set_i((int64)t.scalar<int64>()(0));
    fused_string_split_slice_node.mutable_attr()->insert(
        {"maxsplit", maxsplit_attr});
    // Add node
    Status status;
    Node* node_fused_string_split_slice_node =
        g->AddNode(fused_string_split_slice_node, &status);
    if (status != Status::OK() || !node_fused_string_split_slice_node) {
      LOG(WARNING) << "Add fused_string_split_slice node failed: "
                   << status.error_message();
      return NULL;
    }

    return node_fused_string_split_slice_node;
  }

  virtual bool rebuild_graph(Graph* g, std::vector<const Edge*>& inputs,
                             std::vector<std::vector<const Edge*>>& outputs,
                             Node* node_fused_string_split_slice_node) {
    if (inputs.size() < 5 || outputs.size() > 5) {
      LOG(WARNING) << "Input size[" << inputs.size()
                   << "] is less then 5 or output size[" << outputs.size()
                   << "] is more then 5";
      return false;
    }

    add_iedge(g, node_fused_string_split_slice_node, 0, inputs[0]);
    add_iedge(g, node_fused_string_split_slice_node, 1, inputs[1]);
    add_oedges(g, node_fused_string_split_slice_node, 0, outputs[0]);
    add_oedges(g, node_fused_string_split_slice_node, 1, outputs[1]);
    add_oedges(g, node_fused_string_split_slice_node, 2, outputs[2]);
    add_oedges(g, node_fused_string_split_slice_node, 2, outputs[3]);
    return true;
  }
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_GRAPH_TEMPLATE_STRING_SPLIT_SLICE_BASE_H_
