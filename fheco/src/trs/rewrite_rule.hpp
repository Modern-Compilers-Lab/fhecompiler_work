#pragma once
#include "matching_term.hpp"
#include "program.hpp"
#include "trs_core.hpp"
#include <optional>
#include <unordered_map>

namespace ir
{
class Term;
class Program;
} // namespace ir

namespace fheco_trs
{

class RewriteRule
{
public:
  using matching_term_ir_node_map = std::unordered_map<size_t, std::shared_ptr<ir::Term>>;
  using rhs_factory_function = std::function<MatchingTerm(const ir::Program *prgm, const matching_term_ir_node_map &)>;

private:
  static size_t rule_id;
  size_t id;
  MatchingTerm lhs;
  MatchingTerm static_rhs;
  std::optional<rhs_factory_function> rhs_factory;
  std::optional<MatchingTerm> rewrite_condition;
  bool saving_circuit_flag = false; /* this attribute is to indicate that the rule will be applied only if some
                                     condition satisifed to avoid losing operands if they are already common*/
  /* by definition all variable in rhs must be in lhs */
public:
  RewriteRule() = delete;
  RewriteRule(const RewriteRule &) = default;
  RewriteRule(RewriteRule &&) = default;
  RewriteRule &operator=(const RewriteRule &) = default;
  RewriteRule &operator=(RewriteRule &&) = default;

  // explicit RewriteRule(
  // const MatchingTerm &_lhs, const rhs_factory_function &_rhs_factory, bool _saving_circuit_flag = false);

  // explicit RewriteRule(
  // const MatchingTerm &_lhs, const rhs_factory_function &_rhs_factory, const MatchingTerm &_condition,
  // bool _saving_circuit_flag = false);

  // RewriteRule(const MatchingTerm &_lhs, const rhs_factory_function &_rhs_factory, bool _saving_circuit_flag = false);

  RewriteRule(const MatchingTerm &_lhs, const MatchingTerm &_rhs, bool _saving_circuit_flag = false);

  RewriteRule(
    const MatchingTerm &_lhs, const MatchingTerm &_rhs, const MatchingTerm &_condition,
    bool _saving_circuit_flag = false);

  // RewriteRule(
  // const MatchingTerm &_lhs, const rhs_factory_function &_rhs_factory, const MatchingTerm &_condition,
  // bool _saving_circuit_flag = false);

  size_t get_id() const { return id; }

  const MatchingTerm &get_lhs() const { return lhs; }

  const MatchingTerm &get_static_rhs() const { return static_rhs; }

  MatchingTerm get_rhs(const ir::Program *program, const matching_term_ir_node_map &matching_map) const;

  const std::optional<MatchingTerm> &get_rewrite_condition() const { return rewrite_condition; }

  bool substitute_in_ir(
    std::shared_ptr<ir::Term> ir_node, core::MatchingMap &matching_map, ir::Program *program,
    core::FunctionTable &functions_table) const;

  bool evaluate_rewrite_condition(
    core::MatchingMap &matching_map, ir::Program *program, core::FunctionTable &functions_table) const;

  std::optional<core::MatchingMap> match_with_ir_node(const ir::Program::Ptr &ir_node, ir::Program *program) const;

  ~RewriteRule() {}
};

} // namespace fheco_trs
