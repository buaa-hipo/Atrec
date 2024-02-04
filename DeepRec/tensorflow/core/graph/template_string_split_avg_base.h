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

#ifndef TENSORFLOW_CORE_GRAPH_TEMPLATE_STRING_SPLIT_AVG_BASE_H_
#define TENSORFLOW_CORE_GRAPH_TEMPLATE_STRING_SPLIT_AVG_BASE_H_

#include "tensorflow/core/graph/template_base.h"

namespace tensorflow {

class TemplateStringSplitAvgBase : public TemplateBase {
 public:
  TemplateStringSplitAvgBase() {
    const TempNode n01 = {
      .key = "input_layer/StringSplit_5/StringSplitV2",
      .op = "StringSplitV2",
      .inputs = {/* IteratorGetNext:18 */ "0", 
                 /* input_layer/StringSplit_5/Const */ "1" },
      .outputs = {{"input_layer/sequence_length/strided_slice",
                   "input_layer/sequence_length/strided_slice_1",
                   "input_layer/SparseToDense"},
                  {"input_layer/SparseToDense"},
                  {"input_layer/sequence_length/Shape/Cast",
                   "input_layer/SparseToDense"}}};
    temp_nodes_.emplace_back(n01);

    const TempNode n02 = {
      .key = "input_layer/sequence_length/strided_slice_1",
      .op = "StridedSlice",
      .inputs = {"input_layer/StringSplit_5/StringSplitV2",
                 /* input_layer/sequence_length/strided_slice_1/stack */ "2",
                 /* input_layer/sequence_length/strided_slice_1/stack_1 */ "3",
                 /* input_layer/sequence_length/strided_slice_1/stack_2 */ "4"},
      .outputs = {{"input_layer/sequence_length/ones_like/Shape",
                   "input_layer/sequence_length/add"}}};
    temp_nodes_.emplace_back(n02);

    const TempNode n03 = {
      .key = "input_layer/sequence_length/ones_like/Shape",
      .op = "Shape",
      .inputs = {"input_layer/sequence_length/strided_slice_1"},
      .outputs = {{"input_layer/sequence_length/ones_like"}}};
    temp_nodes_.emplace_back(n03);

    const TempNode n04 = {
      .key = "input_layer/sequence_length/ones_like",
      .op = "Fill",
      .inputs = {"input_layer/sequence_length/ones_like/Shape",
                 /* input_layer/sequence_length/ones_like/Const */ "5"},
      .outputs = {{"input_layer/sequence_length/add"}}};
    temp_nodes_.emplace_back(n04);

    const TempNode n05 = {
      .key = "input_layer/sequence_length/add",
      .op = "AddV2",
      .inputs = {"input_layer/sequence_length/strided_slice_1",
                 "input_layer/sequence_length/ones_like"},
      .outputs = {{"input_layer/sequence_length/SegmentMax"}}};
    temp_nodes_.emplace_back(n05);

    const TempNode n06 = {
      .key = "input_layer/sequence_length/strided_slice",
      .op = "StridedSlice",
      .inputs = {"input_layer/StringSplit_5/StringSplitV2",
                 /* input_layer/sequence_length/strided_slice/stack */ "6",
                 /* input_layer/sequence_length/strided_slice/stack_1 */ "7",
                 /* input_layer/sequence_length/strided_slice/stack_2 */ "8"},
      .outputs = {{"input_layer/sequence_length/SegmentMax"}}};
    temp_nodes_.emplace_back(n06);

    const TempNode n07 = {
      .key = "input_layer/sequence_length/SegmentMax",
      .op = "SegmentMax",
      .inputs = {"input_layer/sequence_length/add",
                 "input_layer/sequence_length/strided_slice"},
      .outputs = {{"input_layer/sequence_length/truediv/Cast"}}};
    temp_nodes_.emplace_back(n07);

    const TempNode n08 = {
      .key = "input_layer/sequence_length/truediv/Cast",
      .op = "Cast",
      .inputs = {"input_layer/sequence_length/SegmentMax"},
      .outputs = {{"input_layer/sequence_length/truediv"}}};
    temp_nodes_.emplace_back(n08);

    const TempNode n09 = {
      .key = "input_layer/sequence_length/truediv/Cast_1",
      .op = "Cast",
      .inputs = {/* input_layer/sequence_length/truediv/y */ "9"},
      .outputs = {{"input_layer/sequence_length/truediv"}}};
    temp_nodes_.emplace_back(n09);

    const TempNode n10 = {
      .key = "input_layer/sequence_length/truediv",
      .op = "RealDiv",
      .inputs = {"input_layer/sequence_length/truediv/Cast",
                 "input_layer/sequence_length/truediv/Cast_1"},
      .outputs = {{"input_layer/sequence_length/Ceil"}}};
    temp_nodes_.emplace_back(n10);

    const TempNode n11 = {
      .key = "input_layer/SparseToDense",
      .op = "SparseToDense",
      .inputs = {"input_layer/StringSplit_5/StringSplitV2",
                 "input_layer/StringSplit_5/StringSplitV2", //:2
                 "input_layer/StringSplit_5/StringSplitV2", //:1
                 /* input_layer/SparseToDense/default_value */ "10"},
      .outputs = {{"input_layer/StringToNumber"}}};
    temp_nodes_.emplace_back(n11);

    const TempNode n12 = {
      .key = "input_layer/sequence_length/Ceil",
      .op = "Ceil",
      .inputs = {"input_layer/sequence_length/truediv"},
      .outputs = {{"input_layer/sequence_length/Cast"}}};
    temp_nodes_.emplace_back(n12);

    const TempNode n13 = {
      .key = "input_layer/sequence_length/Cast",
      .op = "Cast",
      .inputs = {"input_layer/sequence_length/Ceil"},
      .outputs = {{"input_layer/sequence_length/Shape_1",
                   "input_layer/sequence_length"}}};
    temp_nodes_.emplace_back(n13);

    const TempNode n14 = {
      .key = "input_layer/sequence_length/Shape/Cast",
      .op = "Cast",
      .inputs = {"input_layer/StringSplit_5/StringSplitV2"}, //:2
      .outputs = {{"input_layer/sequence_length/strided_slice_2"}}};
    temp_nodes_.emplace_back(n14);

    const TempNode n15 = {
      .key = "input_layer/sequence_length/Shape_1",
      .op = "Shape",
      .inputs = {"input_layer/sequence_length/Cast"},
      .outputs = {{"input_layer/sequence_length/strided_slice_3"}}};
    temp_nodes_.emplace_back(n15);

    const TempNode n16 = {
      .key = "input_layer/sequence_length/strided_slice_2",
      .op = "StridedSlice",
      .inputs = {"input_layer/sequence_length/Shape/Cast",
                 /* input_layer/sequence_length/strided_slice_2/stack */ "11",
                 /* input_layer/sequence_length/strided_slice_2/stack_1 */ "12",
                 /* input_layer/sequence_length/strided_slice_2/stack_2 */ "13"},
      .outputs = {{"input_layer/sequence_length/sub"}}};
    temp_nodes_.emplace_back(n16);

    const TempNode n17 = {
      .key = "input_layer/sequence_length/strided_slice_3",
      .op = "StridedSlice",
      .inputs = {"input_layer/sequence_length/Shape_1",
                 /* input_layer/sequence_length/strided_slice_3/stack */ "14",
                 /* input_layer/sequence_length/strided_slice_3/stack_1 */ "15",
                 /* input_layer/sequence_length/strided_slice_3/stack_2 */ "16"},
      .outputs = {{"input_layer/sequence_length/sub"}}};
    temp_nodes_.emplace_back(n17);

    const TempNode n18 = {
      .key = "input_layer/sequence_length/sub",
      .op = "Sub",
      .inputs = {"input_layer/sequence_length/strided_slice_2",
                 "input_layer/sequence_length/strided_slice_3"},
      .outputs = {{"input_layer/sequence_length/zeros"}}};
    temp_nodes_.emplace_back(n18);

    const TempNode n19 = {
      .key = "input_layer/sequence_length/zeros",
      .op = "Fill",
      .inputs = {"input_layer/sequence_length/sub",
                 /* input_layer/sequence_length/zeros/Const */ "17"},
      .outputs = {{"input_layer/sequence_length"}}};
    temp_nodes_.emplace_back(n19);

    const TempNode n20 = {
      .key = "input_layer/sequence_length",
      .op = "ConcatV2",
      .inputs = {"input_layer/sequence_length/Cast",
                 "input_layer/sequence_length/zeros",
                 /* input_layer/sequence_length/axis */ "18"},
      .outputs = {{"input_layer/ExpandDims"}}};
    temp_nodes_.emplace_back(n20);

    const TempNode n21 = {
      .key = "input_layer/StringToNumber",
      .op = "StringToNumber",
      .inputs = {"input_layer/SparseToDense"},
      .outputs = {{"input_layer/Sum"}}};
    temp_nodes_.emplace_back(n21);

    const TempNode n22 = {
      .key = "input_layer/ExpandDims",
      .op = "ExpandDims",
      .inputs = {"input_layer/sequence_length", 
                 /* input_layer/ExpandDims/dim */ "19"}, 
      .outputs = {{"input_layer/Cast"}}};
    temp_nodes_.emplace_back(n22);

    const TempNode n23 = {
      .key = "input_layer/Sum",
      .op = "Sum",
      .inputs = {"input_layer/StringToNumber",
                 /* input_layer/Sum/reduction_indices */ "20"},
      .outputs = {{"input_layer/truediv"}}};
    temp_nodes_.emplace_back(n23);

    const TempNode n24 = {
      .key = "input_layer/Cast",
      .op = "Cast",
      .inputs = {"input_layer/ExpandDims"},
      .outputs = {{"input_layer/truediv"}}};
    temp_nodes_.emplace_back(n24);

    const TempNode n25 = {
      .key = "input_layer/truediv",
      .op = "RealDiv",
      .inputs = {"input_layer/Sum", 
                 "input_layer/Cast"},
      .outputs = {{/* input_layer/input_layer/price_list/Shape */ "0",
                   /* input_layer/input_layer/price_list/Shape_1 */ "1",
                   /* input_layer/input_layer/price_list/Reshape */ "2",
                   /* linear/linear_model_1/linear_model/price_list/Shape */ "3",
                   /* linear/linear_model_1/linear_model/price_list/Reshape */  "4"}}};
                   
    temp_nodes_.emplace_back(n25);

    first_key_ = "input_layer/StringSplit_5/StringSplitV2";
    num_inputs_ = 21;
    num_outputs_ = 5;
  }

  const string name() { return "string_split_avg_base"; }

  bool add_subgraph(std::map<std::string, MatchedNode>& nodes,
                  std::string name_prefix, Graph* g,
                  std::vector<const Edge*>& inputs,
                  std::vector<std::vector<const Edge*>>& outputs) override {
    LOG(INFO) << "Fusion template[" << name() << "] match op["
              << nodes[first_key_].node->name() << "][new_name:" << name_prefix
              << "_" << name() << "]";

    Node* node_fused_string_split_avg =
        add_fused_string_split_avg_node(nodes, name_prefix, g, inputs, outputs);
    if (!node_fused_string_split_avg) {
      LOG(WARNING) << "Add node_fused_string_split_avg node failed";
      return false;
    }

    return rebuild_graph(g, inputs, outputs, node_fused_string_split_avg);
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
  virtual Node* add_fused_string_split_avg_node(
      std::map<std::string, MatchedNode>& nodes, std::string name_prefix,
      Graph* g, std::vector<const Edge*>& inputs,
      std::vector<std::vector<const Edge*>>& outputs) {
    // construct fused_string_split_avg node
    NodeDef fused_string_split_avg_node;
    add_input(fused_string_split_avg_node, inputs[0]);
    add_input(fused_string_split_avg_node, inputs[1]);
    fused_string_split_avg_node.set_op("StringSplitAvg");
    fused_string_split_avg_node.set_name(name_prefix + name());
    fused_string_split_avg_node.set_device(
        nodes["input_layer/StringSplit_5/StringSplitV2"].node->def().device());
    AttrValue dtype_attr;
    dtype_attr.set_type(DT_FLOAT);
    fused_string_split_avg_node.mutable_attr()->insert({"out_type", dtype_attr});
    AttrValue maxsplit_attr;
    maxsplit_attr.set_i(-1);
    fused_string_split_avg_node.mutable_attr()->insert({"maxsplit", maxsplit_attr});
    // Add node
    Status status;
    Node* node_fused_string_split_avg_node =
        g->AddNode(fused_string_split_avg_node, &status);
    if (status != Status::OK() || !node_fused_string_split_avg_node) {
      LOG(WARNING) << "Add fused_string_split_avg node failed: "
                 << status.error_message();
      return NULL;
    }

    return node_fused_string_split_avg_node;
  }

  virtual bool rebuild_graph(Graph* g, std::vector<const Edge*>& inputs,
                           std::vector<std::vector<const Edge*>>& outputs,
                           Node* node_fused_string_split_avg_node) {
    if (inputs.size() < 2 || outputs.size() > 5) {
      LOG(WARNING) << "Input size[" << inputs.size()
                 << "] is less then 2 or output size[" << outputs.size()
                 << "] is more then 5";
      return false;
    }

    add_iedge(g, node_fused_string_split_avg_node, 0, inputs[0]);
    add_iedge(g, node_fused_string_split_avg_node, 1, inputs[1]);
    add_oedges(g, node_fused_string_split_avg_node, 0, outputs[0]);
    return true;
  }
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_GRAPH_TEMPLATE_STRING_SPLIT_AVG_BASE_H_
