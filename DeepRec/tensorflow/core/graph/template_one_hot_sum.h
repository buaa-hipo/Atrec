// Tianchi bzdjsm

#ifndef TENSORFLOW_CORE_GRAPH_TEMPLATE_ONE_HOT_SUM_H_
#define TENSORFLOW_CORE_GRAPH_TEMPLATE_ONE_HOT_SUM_H_

#include "tensorflow/core/graph/template_base.h"

namespace tensorflow {

class TemplateOneHotSum : public TemplateBase {
 public:
  TemplateOneHotSum() {
    const TempNode n0 = {
      .key = "one_hot_0",
      .op = "OneHot",
      .inputs = {"0","1","2","3"},
      .outputs = {{"sum_0"}}
    };
    temp_nodes_.emplace_back(n0);

    const TempNode n1 = {
      .key = "sum_0",
      .op = "Sum",
      .inputs = {"one_hot_0", "4"},
      .outputs = {{"0"}}
    };
    temp_nodes_.emplace_back(n1);

    first_key_   = "one_hot_0";
    num_inputs_  = 5;
    num_outputs_ = 1;

  }

  const string name() {
    return "one_hot_sum";
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

    Node* node_fused_one_hot_sum = add_fused_one_hot_sum_node(nodes, name_prefix, g, inputs, outputs);
    if (!node_fused_one_hot_sum) {
      LOG(WARNING) << "Add node_fused_one_hot_sum node failed";
      return false;
    }

    return rebuild_graph(g, inputs, outputs, node_fused_one_hot_sum);
  }

  bool CheckConstScalarNode(const NodeDef& node_def, int expected_val) {
    Tensor val;
    Status s = GetNodeAttr(node_def, "value", &val);
    if (val.dtype() != DT_INT32) {
      return false;
    }
    auto v = val.flat<int>();
    return (v.size() == 1) && (v(0) == expected_val);
  }

  bool CheckInputs(std::vector<const Edge*>& inputs) {
    if (inputs.size() != 5) {
      return false;
    }

    if (!CheckConstScalarNode(inputs[4]->src()->def(), -2)) {
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
  
  Node* add_fused_one_hot_sum_node(
      std::map<std::string, MatchedNode>& nodes,
      std::string name_prefix, Graph* g,
      std::vector<const Edge*>& inputs,
      std::vector<std::vector<const Edge*>>& outputs) {
    NodeDef fused_one_hot_sum_node;
    add_input(fused_one_hot_sum_node, inputs[0]);
    add_input(fused_one_hot_sum_node, inputs[1]);
    add_input(fused_one_hot_sum_node, inputs[2]);
    add_input(fused_one_hot_sum_node, inputs[3]);
    fused_one_hot_sum_node.set_op("OneHotSum");
    fused_one_hot_sum_node.set_name(name_prefix + name());
    fused_one_hot_sum_node.set_device(nodes["one_hot_0"].node->def().device());

    DataType T;
    DataType TI;
    int axis;

    Status s = GetNodeAttr(nodes["one_hot_0"].node->def(), "T", &T);
    if (s != Status::OK()){ 
      LOG(WARNING) << "Get one_hot T attr failed: " << s.error_message();
      return NULL;
    }
    s = GetNodeAttr(nodes["one_hot_0"].node->def(), "TI", &TI);
    if (s != Status::OK()){ 
      LOG(WARNING) << "Get one_hot TI attr failed: " << s.error_message();
      return NULL;
    }
    s = GetNodeAttr(nodes["one_hot_0"].node->def(), "axis", &axis);
    if (s != Status::OK()){ 
      LOG(WARNING) << "Get one_hot axis attr failed: " << s.error_message();
      return NULL;
    }

    if (axis != -1) {
      LOG(WARNING) << "/Tianchi bzdjsm/ one_hot axis is not -1, and won't be fused with sum";
      return NULL;
    }

    AttrValue one_hot_sum_T;
    AttrValue one_hot_sum_TI;
    AttrValue one_hot_sum_axis;
    one_hot_sum_T.set_type(T);
    one_hot_sum_TI.set_type(TI);
    one_hot_sum_axis.set_i(axis);
    fused_one_hot_sum_node.mutable_attr()->insert({"T", one_hot_sum_T});
    fused_one_hot_sum_node.mutable_attr()->insert({"TI", one_hot_sum_TI});
    fused_one_hot_sum_node.mutable_attr()->insert({"axis", one_hot_sum_axis});

    // Add node
    Status status;
    Node* node_fused_one_hot_sum_node = g->AddNode(fused_one_hot_sum_node, &status);
    if (status != Status::OK() || !node_fused_one_hot_sum_node) {
      LOG(WARNING) << "/Tianchi bzdjsm/ Add fused_one_hot_sum node failed: " << status.error_message();
      return NULL;
    }

    return node_fused_one_hot_sum_node;
  }

  virtual bool rebuild_graph(Graph* g, std::vector<const Edge*>& inputs,
      std::vector<std::vector<const Edge*>>& outputs,
      Node* node_fused_one_hot_sum_node) {
    if (inputs.size() != 5 || outputs.size() != 1) {
      LOG(WARNING) << "Input size[" << inputs.size() << "] is not 5 or output size["
          << outputs.size() << "] is not 1";
      return false;
    }

    add_iedge(g, node_fused_one_hot_sum_node, 0, inputs[0]);
    add_iedge(g, node_fused_one_hot_sum_node, 1, inputs[1]);
    add_iedge(g, node_fused_one_hot_sum_node, 2, inputs[2]);
    add_iedge(g, node_fused_one_hot_sum_node, 3, inputs[3]);
    add_oedges(g, node_fused_one_hot_sum_node, 0, outputs[0]);
    return true;
  }

};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_GRAPH_TEMPLATE_ONE_HOT_SUM_H_