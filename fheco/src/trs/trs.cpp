#include "trs.hpp"
#include "draw_ir.hpp"
#include "ir_utils.hpp"
#include "trs_const.hpp"
#include <variant>

namespace fheco_trs
{

std::vector<core::MatchingPair> TRS::apply_rule_on_ir_node(
  const std::shared_ptr<ir::Term> &ir_node, RewriteRule &rule, bool &is_rule_applied)
{

  is_rule_applied = false;

  auto matching_map = core::match_ir_node(ir_node, rule.get_lhs());

  if (matching_map != std::nullopt)
  {
    // std::cout << "matched ... \n";
    if (rule.get_rewrite_condition() != std::nullopt)
    {
      // std::cout << "checking condition ...\n";
      if (rule.evaluate_rewrite_condition(*matching_map, program, functions_table))
      {
        is_rule_applied = true;
        auto new_constants = rule.substitute_in_ir(ir_node, *matching_map, program, functions_table);
        return new_constants;
      }
      else
        return {};
    }
    else
    {
      is_rule_applied = true;
      auto new_constants = rule.substitute_in_ir(ir_node, *matching_map, program, functions_table);
      return new_constants;
    }
  }
  else
    return {};
}

void TRS::apply_rules_on_ir_node(const std::shared_ptr<ir::Term> &node, std::vector<RewriteRule> &rules)
{
  size_t curr_rule_index = 0;
  while (curr_rule_index < rules.size())
  {
    /*
      new_nodes_matching_pairs supposed to contain matching pairs for constants only
    */
    bool was_rule_applied = false;
    auto &rule = rules[curr_rule_index];
    do
    {
      auto new_nodes_matching_pairs = apply_rule_on_ir_node(node, rule, was_rule_applied);
      for (auto &matching_pair : new_nodes_matching_pairs)
      {
        if (matching_pair.matching_term.get_value() == std::nullopt)
          throw("only constant are supposed to be inserted as new terms");

        ir::ConstantTableEntry c_table_entry(
          ir::ConstantTableEntryType::constant,
          {matching_pair.ir_node->get_label(), *(matching_pair.matching_term.get_value())});
        program->insert_entry_in_constants_table({matching_pair.ir_node->get_label(), c_table_entry});
      }

    } while (was_rule_applied == true);

    curr_rule_index += 1;
  }
}

void TRS::apply_rewrite_rules_on_program(std::vector<RewriteRule> &rules)
{
  auto &sorted_nodes = program->get_dataflow_sorted_nodes(true);
  for (auto &node : sorted_nodes)
  {
    if (node->get_opcode() == ir::OpCode::undefined)
      continue;

    apply_rules_on_ir_node(node, rules);
  }
}

} // namespace fheco_trs
