// Tianchi bzdjsm

#ifndef TENSORFLOW_CORE_GRAPH_TEMPLATE_SPARSE_STRING_TO_DENSE_NUMBER_H_
#define TENSORFLOW_CORE_GRAPH_TEMPLATE_SPARSE_STRING_TO_DENSE_NUMBER_H_

#include "tensorflow/core/graph/template_base.h"

namespace tensorflow {

class TemplateSparseStringToDenseNumber : public TemplateBase {
 public:
  TemplateSparseStringToDenseNumber() {
    const TempNode n0 = {.key = "to_dense_0",
                         .op = "SparseToDense",
                         .inputs = {"0", "1", "2", "3"},
                         .outputs = {{"to_number_0"}}};
    temp_nodes_.emplace_back(n0);

    const TempNode n1 = {.key = "to_number_0",
                         .op = "StringToNumber",
                         .inputs = {"to_dense_0"},
                         .outputs = {{"0"}}};
    temp_nodes_.emplace_back(n1);

    first_key_ = "to_dense_0";
    num_inputs_ = 4;
    num_outputs_ = 1;
  }

  const string name() { return "sparse_string_to_dense_number"; }

  bool add_subgraph(std::map<std::string, MatchedNode>& nodes,
                    std::string name_prefix, Graph* g,
                    std::vector<const Edge*>& inputs,
                    std::vector<std::vector<const Edge*>>& outputs) override {
    if (!CheckInputs(inputs)) {
      LOG(WARNING) << "Input check failed";
      return false;
    }
    LOG(INFO) << "/Tianchi bzdjsm/ Fusion template[" << name() << "] match op["
              << nodes[first_key_].node->name() << "][new_name:" << name_prefix
              << "_" << name() << "]";

    Node* node_fused_sparse_string_to_dense_number =
        add_fused_sparse_string_to_dense_number_node(nodes, name_prefix, g,
                                                     inputs, outputs);
    if (!node_fused_sparse_string_to_dense_number) {
      LOG(WARNING)
          << "Add node_fused_sparse_string_to_dense_number node failed";
      return false;
    }

    return rebuild_graph(g, inputs, outputs,
                         node_fused_sparse_string_to_dense_number);
  }

  bool CheckInputs(std::vector<const Edge*>& inputs) {
    const NodeDef& default_value = inputs[3]->src()->def();
    Tensor val;
    Status s = GetNodeAttr(default_value, "value", &val);
    return (val.dtype() == DT_STRING) && (val.flat<tstring>()(0) == "0");
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
  Node* add_fused_sparse_string_to_dense_number_node(
      std::map<std::string, MatchedNode>& nodes, std::string name_prefix,
      Graph* g, std::vector<const Edge*>& inputs,
      std::vector<std::vector<const Edge*>>& outputs) {
    NodeDef fused_sparse_string_to_dense_number_node;
    add_input(fused_sparse_string_to_dense_number_node, inputs[0]);
    add_input(fused_sparse_string_to_dense_number_node, inputs[1]);
    add_input(fused_sparse_string_to_dense_number_node, inputs[2]);
    fused_sparse_string_to_dense_number_node.set_op(
        "SparseStringToDenseNumber");
    fused_sparse_string_to_dense_number_node.set_name(name_prefix + name());
    fused_sparse_string_to_dense_number_node.set_device(
        nodes["to_dense_0"].node->def().device());

    DataType T;
    DataType Tindices;
    bool validate_indices;
    DataType out_type;

    Status s = GetNodeAttr(nodes["to_dense_0"].node->def(), "T", &T);
    if (s != Status::OK()){ 
      LOG(WARNING) << "Failed: " << s.error_message();
      return NULL;
    }
    s = GetNodeAttr(nodes["to_dense_0"].node->def(), "Tindices", &Tindices);
    if (s != Status::OK()){ 
      LOG(WARNING) << "Failed: " << s.error_message();
      return NULL;
    }
    s = GetNodeAttr(nodes["to_dense_0"].node->def(), "validate_indices",
                    &validate_indices);
    if (s != Status::OK()){ 
      LOG(WARNING) << "Failed: " << s.error_message();
      return NULL;
    }
    s = GetNodeAttr(nodes["to_number_0"].node->def(), "out_type", &out_type);
    if (s != Status::OK()){ 
      LOG(WARNING) << "Failed: " << s.error_message();
      return NULL;
    }

    AttrValue attr_T;
    AttrValue attr_Tindices;
    AttrValue attr_validate_indices;
    AttrValue attr_out_type;
    attr_T.set_type(T);
    attr_Tindices.set_type(Tindices);
    attr_validate_indices.set_b(validate_indices);
    attr_out_type.set_type(out_type);

    fused_sparse_string_to_dense_number_node.mutable_attr()->insert(
        {"T", attr_T});
    fused_sparse_string_to_dense_number_node.mutable_attr()->insert(
        {"Tindices", attr_Tindices});
    fused_sparse_string_to_dense_number_node.mutable_attr()->insert(
        {"validate_indices", attr_validate_indices});
    fused_sparse_string_to_dense_number_node.mutable_attr()->insert(
        {"out_type", attr_out_type});

    // Add node
    Status status;
    Node* node_fused_sparse_string_to_dense_number_node =
        g->AddNode(fused_sparse_string_to_dense_number_node, &status);
    if (status != Status::OK() ||
        !node_fused_sparse_string_to_dense_number_node) {
      LOG(WARNING) << "/Tianchi bzdjsm/ Add "
                      "fused_sparse_string_to_dense_number node failed: "
                   << status.error_message();
      return NULL;
    }

    return node_fused_sparse_string_to_dense_number_node;
  }

  virtual bool rebuild_graph(
      Graph* g, std::vector<const Edge*>& inputs,
      std::vector<std::vector<const Edge*>>& outputs,
      Node* node_fused_sparse_string_to_dense_number_node) {
    if (inputs.size() != 4 || outputs.size() != 1) {
      LOG(WARNING) << "Input size[" << inputs.size()
                   << "] is not 5 or output size[" << outputs.size()
                   << "] is not 1";
      return false;
    }

    add_iedge(g, node_fused_sparse_string_to_dense_number_node, 0, inputs[0]);
    add_iedge(g, node_fused_sparse_string_to_dense_number_node, 1, inputs[1]);
    add_iedge(g, node_fused_sparse_string_to_dense_number_node, 2, inputs[2]);
    add_oedges(g, node_fused_sparse_string_to_dense_number_node, 0, outputs[0]);
    return true;
  }
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_GRAPH_TEMPLATE_SPARSE_STRING_TO_DENSE_NUMBER_H_