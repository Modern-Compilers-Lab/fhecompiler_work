#include "fheco/ir/func.hpp"
#include "fheco/ir/term.hpp"
#include "fheco/util/common.hpp"
#include "fheco/util/draw_ir.hpp"
#include "fheco/util/draw_ir_util.hpp"
#include <cstddef>
#include <stack>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

using namespace std;

namespace fheco::util
{
void draw_ir(const shared_ptr<ir::Func> &func, ostream &os, bool id_as_label, bool show_key, bool impose_operands_order)
{
  os << "digraph \"" << func->name() << "\" {\n";
  os << "node [shape=circle width=1 margin=0]\n";
  os << "edge [dir=back]\n";

  for (auto term : func->get_top_sorted_terms())
  {
    os << term->id() << " [" << make_term_attrs(func, term, id_as_label) << "]\n";
    if (!term->is_operation())
      continue;

    const auto &operands = term->operands();
    for (auto operand : operands)
      os << term->id() << " -> " << operand->id() << '\n';

    // try to impose operands order
    if (impose_operands_order && operands.size() > 1)
    {
      for (size_t i = 0; i < operands.size() - 1; ++i)
        os << operands[i]->id() << " -> ";
      os << operands.back()->id() << " [style=invis]\n";
    }
  }

  if (show_key)
    os << terms_key;

  os << "}\n";
}

void draw_term(
  const std::shared_ptr<ir::Func> &func, const ir::Term *term, int depth, ostream &os, bool id_as_label, bool show_key,
  bool impose_operands_order)
{
  os << "digraph \"" << func->name() << "\" {\n";
  os << "node [shape=circle width=1 margin=0]\n";
  os << "edge [dir=back]\n";

  struct Call
  {
    const ir::Term *term_;
    int depth_;
    bool children_processed_;
  };
  stack<Call> call_stack;

  unordered_set<size_t> visited_terms_ids;
  call_stack.push(Call{term, depth, false});
  while (!call_stack.empty())
  {
    auto top_call = call_stack.top();
    call_stack.pop();
    auto top_term = top_call.term_;
    if (top_call.children_processed_)
    {
      visited_terms_ids.insert(top_term->id());

      os << top_term->id() << " [" << make_term_attrs(func, top_term, id_as_label) << "]\n";
      if (!top_term->is_operation())
        continue;

      const auto &operands = top_term->operands();
      for (auto operand : operands)
        os << top_term->id() << " -> " << operand->id() << '\n';

      // try to impose operands order
      if (impose_operands_order && operands.size() > 1)
      {
        for (size_t i = 0; i < operands.size() - 1; ++i)
          os << operands[i]->id() << " -> ";
        os << operands.back()->id() << " [style=invis]\n";
      }
      continue;
    }

    if (auto it = visited_terms_ids.find(top_term->id()); it != visited_terms_ids.end())
      continue;

    if (top_call.depth_ <= 0)
    {
      os << top_term->id() << " [" << make_term_attrs(func, top_term, true) << "]\n";
      continue;
    }

    call_stack.push(Call{top_term, top_call.depth_, true});
    for (auto it = top_term->operands().crbegin(); it != top_term->operands().crend(); ++it)
      call_stack.push(Call{*it, top_call.depth_ - 1, false});
  }
  if (show_key)
    os << terms_key;

  os << "}\n";
}

void draw_term_matcher(const trs::TermMatcher &term_matcher, ostream &os, bool show_key, bool impose_operands_order)
{
  os << "digraph \"$" << term_matcher.id() << "\" {\n";
  os << "node [shape=circle width=1 margin=0]\n";
  os << "edge [dir=none]\n";

  int leaf_occ_id = 0;
  draw_term_matcher_util(term_matcher, os, impose_operands_order, leaf_occ_id);

  if (show_key)
    os << term_matchers_key;

  os << "}\n";
}

string make_term_attrs(const shared_ptr<ir::Func> &func, const ir::Term *term, bool id_as_label)
{
  string attrs = "label=\"" + make_term_label(func, term, id_as_label) + "\"";
  auto type_attrs = term_type_to_attrs.at(term->type());
  if (type_attrs.size())
    attrs += " " + type_attrs;
  auto qualif_attrs = term_qualifs_to_attrs.at(func->data_flow().get_qualif(term));
  if (qualif_attrs.size())
    attrs += " " + qualif_attrs;
  return attrs;
}

string make_term_label(const shared_ptr<ir::Func> &func, const ir::Term *term, bool id_as_label)
{
  if (id_as_label)
    return "$" + to_string(term->id());

  // operation term
  if (term->is_operation())
  {
    if (auto output_info = func->data_flow().get_output_info(term); output_info)
      return term->op_code().str_repr() + " (" + output_info->label_ + ")";

    return term->op_code().str_repr();
  }

  // leaf term
  if (auto input_info = func->data_flow().get_input_info(term); input_info)
  {
    string label = input_info->label_;
    if (auto output_info = func->data_flow().get_output_info(term); output_info)
      label += "(" + output_info->label_ + ")";
    return label;
  }
  else if (auto const_info = func->data_flow().get_const_info(term); const_info)
  {
    if (auto output_info = func->data_flow().get_output_info(term); output_info)
      return output_info->label_;

    if (const_info->is_scalar_)
      return to_string(const_info->val_[0]);

    return "const_$" + to_string(term->id());
  }
  else
    throw logic_error("invalid leaf term, non-input and non-const");
}

void draw_term_matcher_util(
  const trs::TermMatcher &term_matcher, ostream &os, bool impose_operands_order, int &leaf_occ_id)
{
  if (term_matcher.is_leaf())
  {
    os << term_matcher.id() << "." << leaf_occ_id << " [label=\"";
    ++leaf_occ_id;
    if (term_matcher.val())
    {
      auto val = *term_matcher.val();
      if (is_scalar(val))
        os << val[0];
      else
        os << "$val" << term_matcher.id();
    }
    else if (term_matcher.label())
      os << *term_matcher.label();
    else
      throw invalid_argument("variable term_matcher without label");

    os << "\" " << term_matcher_type_to_attrs.at(term_matcher.type()) << "]\n";
    return;
  }

  vector<int> term_leaves_occ_ids;
  const auto &operands = term_matcher.operands();
  for (const auto &operand : operands)
  {
    if (operand.is_leaf())
      term_leaves_occ_ids.push_back(leaf_occ_id);
    draw_term_matcher_util(operand, os, impose_operands_order, leaf_occ_id);
  }

  vector<int> op_gen_leaves_occ_ids;
  for (const auto &op_gen : term_matcher.op_code().generators())
  {
    if (op_gen.is_leaf())
      op_gen_leaves_occ_ids.push_back(leaf_occ_id);
    draw_op_gen_matcher_util(op_gen, os, impose_operands_order, leaf_occ_id);
  }

  os << term_matcher.id() << " [label=\"" << term_matcher.op_code() << "\" "
     << term_matcher_type_to_attrs.at(term_matcher.type()) << "]\n";

  int leaf_idx = 0;
  for (const auto &op_gen : term_matcher.op_code().generators())
  {
    os << term_matcher.id() << " -> -" << op_gen.id();
    if (op_gen.is_leaf())
    {
      os << "." << op_gen_leaves_occ_ids[leaf_idx];
      ++leaf_idx;
    }
    os << '\n';
  }

  leaf_idx = 0;
  for (const auto &operand : operands)
  {
    os << term_matcher.id() << " -> " << operand.id();
    if (operand.is_leaf())
    {
      os << "." << term_leaves_occ_ids[leaf_idx];
      ++leaf_idx;
    }
    os << '\n';
  }

  // try to impose operands order
  if (impose_operands_order && operands.size() > 1)
  {
    leaf_idx = 0;
    for (size_t i = 0; i < operands.size() - 1; ++i)
    {
      os << operands[i].id();
      if (operands[i].is_leaf())
      {
        os << "." << term_leaves_occ_ids[leaf_idx];
        ++leaf_idx;
      }
      os << " -> ";
    }
    os << operands.back().id();
    if (operands.back().is_leaf())
    {
      os << "." << term_leaves_occ_ids[leaf_idx];
      ++leaf_idx;
    }
    os << " [style=invis]\n";
  }
}

void draw_op_gen_matcher_util(
  const trs::OpGenMatcher &op_gen_matcher, ostream &os, bool impose_operands_order, int &leaf_occ_id)
{
  if (op_gen_matcher.is_leaf())
  {
    os << "-" << op_gen_matcher.id() << "." << leaf_occ_id << " [label=\"";
    ++leaf_occ_id;
    if (op_gen_matcher.val())
      os << *op_gen_matcher.val();
    else if (op_gen_matcher.label())
      os << *op_gen_matcher.label();
    else
      throw invalid_argument("variable op_gen_matcher without label");

    os << "\"]\n";
    return;
  }

  vector<int> leaf_occ_ids;
  const auto &operands = op_gen_matcher.operands();
  for (const auto &operand : operands)
  {
    if (operand.is_leaf())
      leaf_occ_ids.push_back(leaf_occ_id);
    draw_op_gen_matcher_util(operand, os, impose_operands_order, leaf_occ_id);
  }

  os << "-" << op_gen_matcher.id() << " [label=\"" << op_gen_matcher.op_code() << "\"]\n";

  int leaf_idx = 0;
  for (const auto &operand : operands)
  {
    os << "-" << op_gen_matcher.id() << " -> -" << operand.id();
    if (operand.is_leaf())
    {
      os << "." << leaf_occ_ids[leaf_idx];
      ++leaf_idx;
    }
    os << '\n';
  }

  // try to impose operands order
  if (impose_operands_order && operands.size() > 1)
  {
    leaf_idx = 0;
    for (size_t i = 0; i < operands.size() - 1; ++i)
    {
      os << "-" << operands[i].id();
      if (operands[i].is_leaf())
      {
        os << "." << leaf_occ_ids[leaf_idx];
        ++leaf_idx;
      }
      os << " -> ";
    }
    os << "-" << operands.back().id();
    if (operands.back().is_leaf())
    {
      os << "." << leaf_occ_ids[leaf_idx];
      ++leaf_idx;
    }
    os << " [style=invis]\n";
  }
}
} // namespace fheco::util
