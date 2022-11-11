#include "term.hpp"
#include <unordered_set>

namespace ir
{

void Term::delete_operand_term(const std::string &term_label) {}

void Term::insert_parent_label(const std::string &label)
{
  parents_labels.insert(label);
}

void Term::add_operand(const Ptr &operand)
{
  (*operation_attribute).operands.push_back(operand);
}

/*

  This function merges a given node that represents a term with another node that represents another term, the other
  node needs to be an operation node excluding assignement and encryption

*/

bool Term::merge_with_node(Ptr node_to_merge_with)
{
  /*
    nullptr edge case
  */
  if (node_to_merge_with == nullptr)
    return false;

  /*
    merge a node with itself edge case
  */
  if (node_to_merge_with->get_label() == this->get_label())
    return false;

  /*
    node to merge with is not an operation node
  */

  if (!node_to_merge_with->is_operation_node())
    return false;

  /*
    a constant node cant be merged with any other node
  */
  if (operation_attribute == std::nullopt)
    return false;

  /*
    node is operation but it represents an assignement or encryption, encoding ..
  */
  const std::unordered_set<ir::OpCode> static_opcodes = {
    ir::OpCode::assign, ir::OpCode::encrypt}; // don't merge with these operations

  ir::OpCode node_opcode = node_to_merge_with->get_opcode();

  if (static_opcodes.find(node_opcode) != static_opcodes.end())
    return false;

  /*
    two nodes must have same operation code, except for case of assign for the node to merge
  */
  if ((*operation_attribute).opcode != node_to_merge_with->get_opcode())
    return false;

  /*
    we dont merge outputs
  */

  /*
    at this point we need to merge, but we first we need to decide how and when we merge also if there is any specific
    data structure is needed for performance
  */

  auto mergening_condition = [&](const Ptr &term_ptr) -> bool {
    // this condition will have an impact on the quality of optimizations
    return term_ptr->get_parents_labels().size() == 1;
  };

  if (mergening_condition(node_to_merge_with))
  {
    /*
      Mergening steps are simple. Having two nodes n1 and n2 where we want to merge them, in that n1 will contain
      operands of n2 and n2 will be deleted. mergening is done from lower to higher level.
    */
    auto operands_to_merge_with_list = node_to_merge_with->get_operands();
    auto &operands_list = (*operation_attribute).operands;
    // operands_list.replace_node_with_list(node_to_merge_with->get_label(), operands_to_merge_with_list);
  }

  return true;
}

} // namespace ir
