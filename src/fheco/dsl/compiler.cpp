#include "fheco/dsl/compiler.hpp"
#include "fheco/trs/trs.hpp"
#include "fheco/passes/passes.hpp"
#include "fheco/util/common.hpp"
#include <iostream>
#include <stdexcept>
#include <unordered_set>
#include <utility>

using namespace std;

namespace fheco
{
Compiler::FuncsTable Compiler::funcs_table_{};

Compiler::FuncsTable::const_iterator Compiler::active_func_it_ = Compiler::funcs_table_.cend();

bool Compiler::cse_enabled_ = true;

bool Compiler::order_operands_enabled_ = true;

bool Compiler::const_folding_enabled_ = true;

bool Compiler::scalar_vector_shape_ = true;

void Compiler::compile(
  shared_ptr<ir::Func> func, Ruleset ruleset, trs::RewriteHeuristic rewrite_heuristic, int64_t max_iter,
  bool rewrite_created_sub_terms)
{
  // clog << "remove_dead_code\n";
  // func->remove_dead_code();

  switch (ruleset)
  {
  case Ruleset::depth:
  {
    clog << "depth_trs\n";
    trs::TRS depth_trs{func, trs::Ruleset::depth_ruleset(func)};
    depth_trs.run(rewrite_heuristic, max_iter, false, rewrite_created_sub_terms);
    break;
  }

  case Ruleset::ops_cost:
  {
    clog << "ops_cost_trs\n";
    trs::TRS ops_cost_trs{func, trs::Ruleset::ops_cost_ruleset(func)};
    ops_cost_trs.run(rewrite_heuristic, max_iter, false, rewrite_created_sub_terms);
    break;
  }

  case Ruleset::joined:
  {
    clog << "joined_trs\n";
    trs::TRS joined_trs{func, trs::Ruleset::joined_ruleset(func)};
    joined_trs.run(rewrite_heuristic, max_iter, false, rewrite_created_sub_terms);
    break;
  }

  default:
    throw invalid_argument("invalid ruleset selector");
    break;
  }

  // clog << "convert_scalar_mul_to_add\n";
  // passes::convert_scalar_mul_to_add(func, 1 << 30);

  // clog << "reduce_rotation_keys\n";
  // unordered_set<int> rotation_steps_keys;
  // rotation_steps_keys =
  //   passes::reduce_rotation_keys(func, 2 * util::get_power_of_two(util::next_power_of_two(func->slot_count())));

  // clog << "insert_relin_ops\n";
  // size_t relin_keys_count = passes::lazy_relin_heuristic(func, 3);
}

void Compiler::add_func(shared_ptr<ir::Func> func)
{
  if (auto it = funcs_table_.find(func->name()); it != funcs_table_.end())
    throw invalid_argument("function with this name already exists");

  active_func_it_ = funcs_table_.emplace(func->name(), move(func)).first;
}

const shared_ptr<ir::Func> &Compiler::get_func(const string &name)
{
  auto it = funcs_table_.find(name);
  if (it == funcs_table_.end())
    throw invalid_argument("no function with this name was found");

  return it->second;
}

void Compiler::set_active_func(const string &name)
{
  active_func_it_ = funcs_table_.find(name);
  if (active_func_it_ == funcs_table_.cend())
    throw invalid_argument("no function with this name was found");
}

void Compiler::delete_func(const string &name)
{
  if (active_func()->name() == name)
    active_func_it_ = funcs_table_.end();
  funcs_table_.erase(name);
}
} // namespace fheco

namespace std
{
ostream &operator<<(ostream &os, fheco::Compiler::Ruleset ruleset)
{
  switch (ruleset)
  {
  case fheco::Compiler::Ruleset::depth:
    os << "depth";
    break;

  case fheco::Compiler::Ruleset::ops_cost:
    os << "ops_cost";
    break;

  case fheco::Compiler::Ruleset::joined:
    os << "joined";
    break;

  default:
    throw invalid_argument("invalid ruleset selector");
    break;
  }

  return os;
}
} // namespace std
