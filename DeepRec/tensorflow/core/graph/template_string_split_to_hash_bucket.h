// Tianchi bzdjsm

#ifndef TENSORFLOW_CORE_GRAPH_TEMPLATE_STRING_SPLIT_TO_HASH_BUCKET_H_
#define TENSORFLOW_CORE_GRAPH_TEMPLATE_STRING_SPLIT_TO_HASH_BUCKET_H_

#include "tensorflow/core/graph/template_base.h"

namespace tensorflow {

class TemplateStringSplitToHashBucket: public TemplateBase {
 public:
  TemplateStringSplitToHashBucket() {

    // TempNode 的含义是 TemplateNode
    // key 的用途是放到一个 map 中时作为这个 node 的键使用
    temp_nodes_.push_back(TempNode{
      .key = "string_split_v2_op",
      .op = "StringSplitV2",  // 这里的 op 会用于与图中的 node->type_string() 对比做匹配
      .inputs = {"0", "1"},   // "0" 是输入 string, "1" 是分隔符 sep
      .outputs = {            // 每个输出可以连接多个边, 因此是 vector of vector
        {"0"},
        {"string_to_hash_bucket_fast"},  // values
        {"2"}}
    });

    temp_nodes_.push_back(TempNode{
      .key = "string_to_hash_bucket_fast",
      .op = "StringToHashBucketFast",
      .inputs = {"string_split_v2_op"},
      .outputs = {{"1"}}
    });

    first_key_ = "string_split_v2_op";
    num_inputs_ = 2;
    num_outputs_ = 3;
  }

  const string name() {
    return "string_split_to_hash_bucket";
  }

  bool add_subgraph(std::map<std::string, MatchedNode>& nodes,
      std::string name_prefix, Graph* g,
      std::vector<const Edge*>& inputs,
      std::vector<std::vector<const Edge*>>& outputs) override {

    LOG(INFO) << "Fusion template[" << name() << "] match op[" << nodes[first_key_].node->name() <<
        "][new_name:" << name_prefix << "_" << name() << "]";

    Node* node_fused_op = add_fused_op(nodes, name_prefix, g, inputs, outputs);
    if (!node_fused_op) {
      LOG(WARNING) << "Add fused_op failed";
      return false;
    }

    return rebuild_graph(g, inputs, outputs, node_fused_op);
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
  virtual Node* add_fused_op(
      std::map<std::string, MatchedNode>& nodes,
      std::string name_prefix, Graph* g,
      std::vector<const Edge*>& inputs,
      std::vector<std::vector<const Edge*>>& outputs) {

    // construct string split to number and mean op
    NodeDef def_fused_op;
    def_fused_op.set_op("StringSplitToHashBucket");
    def_fused_op.set_name(name_prefix + "_" + name());
    auto string_to_hash_bucket_fast_op_attr = nodes["string_to_hash_bucket_fast"].node->def().attr();
    def_fused_op.mutable_attr()->insert({"num_buckets", string_to_hash_bucket_fast_op_attr.at("num_buckets")});
    auto string_split_v2_attr = nodes["string_split_v2_op"].node->def().attr();
    def_fused_op.mutable_attr()->insert({"maxsplit", string_split_v2_attr.at("maxsplit")});

    if (inputs.size() >= 2) {
      add_input(def_fused_op, inputs[0]);
      add_input(def_fused_op, inputs[1]);
    } else {
      LOG(WARNING) << "Input size[" << inputs.size() << "] is less than 2";
      return NULL;
    }

    // Add node
    Status status;
    Node* node_fused_op_add = g->AddNode(def_fused_op, &status);
    if (status != Status::OK()) {
      LOG(WARNING) << "Add node failed: " << status.error_message();
      return NULL;
    }

    return node_fused_op_add;
  }

  virtual bool rebuild_graph(Graph* g, std::vector<const Edge*>& inputs,
      std::vector<std::vector<const Edge*>>& outputs,
      Node* node_fused_op) {

    if (inputs.size() < 2 || outputs.size() < 3) {
      LOG(WARNING) << "Input size[" << inputs.size() << "] is less than 2 or output size["
          << outputs.size() << "] > 1";
      return false;
    }

    add_iedge(g, node_fused_op, 0, inputs[0]);
    add_iedge(g, node_fused_op, 1, inputs[1]);

    add_oedges(g, node_fused_op, 0, outputs[0]);
    add_oedges(g, node_fused_op, 1, outputs[1]);
    add_oedges(g, node_fused_op, 2, outputs[2]);

    return true;
  }

};  // class TemplateStringSplitToHashBucket

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_GRAPH_TEMPLATE_STRING_SPLIT_TO_HASH_BUCKET_H_