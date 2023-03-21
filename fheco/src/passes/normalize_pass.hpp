#pragma once

#include "program.hpp"
#include "trs.hpp"
#include <unordered_set>

namespace fheco_passes
{

class Normalizer
{
private:
  ir::Program *program;

  std::unordered_set<ir::OpCode> target_arithmetic_opcodes = {ir::OpCode::add, ir::OpCode::mul, ir::OpCode::sub};

  /* this set contains nodes that shouldn't be considered for any transformation, so transformations are done locally
   * for these expressions */
  std::unordered_set<ir::Term::Ptr> consolidated_exprs;

  void normalize_rotations(const ir::Term::Ptr &node);

  void flatten_by_one_level(const ir::Term::Ptr &node);

  void recover_binary_form(ir::Term::Ptr &node);

  ir::Term::Ptr recover_binary_form_helper(
    const ir::Term::Ptr &origin_node, const std::vector<ir::Term::Ptr> &operands, size_t start_index, size_t end_index);

public:
  Normalizer(ir::Program *prgm) : program(prgm) {}
  void normalize();
};

} // namespace fheco_passes
