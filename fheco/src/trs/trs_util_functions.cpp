#include "trs_util_functions.hpp"
#include "ir_utils.hpp"

namespace fheco_trs
{
namespace util_functions
{

  MatchingTerm fold(
    MatchingTerm term_to_fold, std::unordered_map<size_t, ir::Program::Ptr> &matching_map, ir::Program *program)
  {
    ir::Program::Ptr folded_ir_node = ir::fold_ir_term(matching_map[term_to_fold.get_term_id()], program);
    matching_map[term_to_fold.get_term_id()] = folded_ir_node;

    return term_to_fold;
  }

  MatchingTerm get_opcode(
    MatchingTerm term, std::unordered_map<size_t, ir::Program::Ptr> &matching_map, ir::Program *program)
  {
    auto it = matching_map.find(term.get_term_id());
    if (it == matching_map.end())
      throw("term doesnt exist in matching_map in get_opcode");

    // comparison needs to be done with ir::OpCode and not fheco_trs::OpCode

    return static_cast<int>(it->second->get_opcode());
  }

} // namespace util_functions

} // namespace fheco_trs