// Tianchi bzdjsm

#ifndef TENSORFLOW_CORE_GRAPH_TEMPLATE_CONCAT_REDUCE_SUM_H_
#define TENSORFLOW_CORE_GRAPH_TEMPLATE_CONCAT_REDUCE_SUM_H_

#include "tensorflow/core/graph/template_base.h"

namespace tensorflow {

class TemplateConcatReduceSum : public TemplateBase {
 public:
  TemplateConcatReduceSum() {
    const TempNode n0 = {
      .key = "concat_0",
      .op = "ConcatV2",
      .inputs = {"0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17",
      "18","19","20","21","22","23","24","25","26","27","28","29"},
      .outputs = {{"sum_0"}}
    };
    temp_nodes_.emplace_back(n0);

    const TempNode n1 = {
      .key = "sum_0",
      .op = "Sum",
      .inputs = {"concat_0", "30"},
      .outputs = {{"0"}}
    };
    temp_nodes_.emplace_back(n1);

    first_key_   = "concat_0";
    num_inputs_  = 31;
    num_outputs_ = 1;

  }

  const string name() {
    return "concat_reduce_sum";
  }

  bool add_subgraph(std::map<std::string, MatchedNode>& nodes,
    std::string name_prefix, Graph* g,
    std::vector<const Edge*>& inputs,
    std::vector<std::vector<const Edge*>>& outputs) override {
    if (!CheckInputs(inputs)) {
      LOG(WARNING) << "Input check failed";
      return false;
    }
    LOG(INFO) << "/Tianchi bzdjsm/ Fusion template[" << name() << "] match op[" << nodes[first_key_].node->name() <<
          "][new_name:" << name_prefix << "_" << name() << "]";

    Node* node_fused_concat_reduce_sum = add_fused_concat_reduce_sum_node(nodes, name_prefix, g, inputs, outputs);
    if (!node_fused_concat_reduce_sum) {
      LOG(WARNING) << "Add node_fused_concat_reduce_sum node failed";
      return false;
    }

    return rebuild_graph(g, inputs, outputs, node_fused_concat_reduce_sum);
  }


  bool CheckInputs(std::vector<const Edge*>& inputs) {
    if (inputs.size() != 31) {
      return false;
    }
    return true;
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
  
  Node* add_fused_concat_reduce_sum_node(
      std::map<std::string, MatchedNode>& nodes,
      std::string name_prefix, Graph* g,
      std::vector<const Edge*>& inputs,
      std::vector<std::vector<const Edge*>>& outputs) {
    NodeDef fused_concat_reduce_sum_node;
    for (int i = 0; i < 29; i++){
      add_input(fused_concat_reduce_sum_node, inputs[i]);
    }

    fused_concat_reduce_sum_node.set_op("ConcatReduceSum");
    fused_concat_reduce_sum_node.set_name(name_prefix + name());
    fused_concat_reduce_sum_node.set_device(nodes["concat_0"].node->def().device());

    int N;
    DataType T;

    Status s = GetNodeAttr(nodes["concat_0"].node->def(), "N", &N);
    if (s != Status::OK()){ 
      LOG(WARNING) << "Get concat N attr failed: " << s.error_message();
      while (true);
    }
    s = GetNodeAttr(nodes["concat_0"].node->def(), "T", &T);
    if (s != Status::OK()){ 
      LOG(WARNING) << "Get concat T attr failed: " << s.error_message();
      while (true);
    }

    AttrValue concat_reduce_sum_N;
    AttrValue concat_reduce_sum_T;
    concat_reduce_sum_N.set_i(N);
    concat_reduce_sum_T.set_type(T);

    fused_concat_reduce_sum_node.mutable_attr()->insert({"N", concat_reduce_sum_N});
    fused_concat_reduce_sum_node.mutable_attr()->insert({"T", concat_reduce_sum_T});

    // Add node
    Status status;
    Node* node_fused_concat_reduce_sum_node = g->AddNode(fused_concat_reduce_sum_node, &status);
    if (status != Status::OK() || !node_fused_concat_reduce_sum_node) {
      LOG(WARNING) << "/Tianchi bzdjsm/ Add fused_concat_reduce_sum node failed: " << status.error_message();
      return NULL;
    }

    return node_fused_concat_reduce_sum_node;
  }

  virtual bool rebuild_graph(Graph* g, std::vector<const Edge*>& inputs,
      std::vector<std::vector<const Edge*>>& outputs,
      Node* node_fused_concat_reduce_sum_node) {
    if (inputs.size() != 31 || outputs.size() != 1) {
      LOG(WARNING) << "Input size[" << inputs.size() << "] is not 31 or output size["
          << outputs.size() << "] is not 1";
      return false;
    }
    for (int i = 0; i < 29; i++){
      add_iedge(g, node_fused_concat_reduce_sum_node, i, inputs[i]);
    }
    add_oedges(g, node_fused_concat_reduce_sum_node, 0, outputs[0]);
    return true;
  }

};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_GRAPH_TEMPLATE_CONCAT_REDUCE_SUM_H_