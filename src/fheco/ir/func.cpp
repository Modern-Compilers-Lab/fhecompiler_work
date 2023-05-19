#include "fheco/ir/func.hpp"
#include "fheco/dsl/ciphertext.hpp"
#include "fheco/dsl/compiler.hpp"
#include "fheco/dsl/plaintext.hpp"
#include "fheco/util/common.hpp"
#include <iostream>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

using namespace std;

namespace fheco::ir
{
Func::Func(string name, size_t slot_count, integer modulus, bool signedness, bool delayed_reduction)
  : name_{move(name)}, slot_count_{slot_count}, need_cyclic_rotations_{false},
    clear_data_evaluator_{slot_count_, modulus, signedness, delayed_reduction}
{
  if (!is_valid_slot_count(slot_count_))
    throw invalid_argument("slot_count must be a power of two");
}

Func::Func(string name, size_t slot_count, int bit_width, bool signedness)
  : Func(move(name), slot_count, (2 << (bit_width - 1)) - 1, signedness, true)
{}

bool Func::is_valid_slot_count(size_t slot_count)
{
  return slot_count != 0 && (slot_count & (slot_count - 1)) == 0;
}

template <typename T>
void Func::init_input(T &input, string label)
{
  Term::Type term_type;
  if constexpr (is_same<T, Ciphertext>::value)
    term_type = Term::Type::cipher;
  else
    term_type = Term::Type::plain;
  input.id_ = data_flow_.insert_input(term_type, ParamTermInfo{move(label), input.example_val_})->id();
}

template <typename T>
void Func::init_const(T &constant, PackedVal packed_val)
{
  constant.id_ = data_flow_.insert_const(move(packed_val))->id();
}

template <typename TArg, typename TDest>
void Func::operate_unary(OpCode op_code, const TArg &arg, TDest &dest)
{
  auto arg_term = data_flow_.find_term(arg.id());
  if (!arg_term)
    throw invalid_argument("operand not defined");

  dest.shape_ = arg.shape();

  if (arg.example_val())
  {
    if constexpr (is_same<TArg, Plaintext>::value && is_same<TDest, Ciphertext>::value)
    {
      if (op_code.type() != OpCode::Type::encrypt)
        throw logic_error("expected encrypt plaintext");

      dest.example_val_ = *arg.example_val();
    }
    else
    {
      PackedVal dest_example_val;
      clear_data_evaluator_.operate_unary(op_code, *arg.example_val(), dest_example_val);
      dest.example_val_ = move(dest_example_val);
    }
  }

  vector<Term *> operands{arg_term};
  dest.id_ = insert_op_term(move(op_code), move(operands))->id();
}

template <typename TArg1, typename TArg2, typename TDest>
void Func::operate_binary(OpCode op_code, const TArg1 &arg1, const TArg2 &arg2, TDest &dest)
{
  auto arg1_term = data_flow_.find_term(arg1.id());
  auto arg2_term = data_flow_.find_term(arg2.id());
  if (!arg1_term || !arg2_term)
    throw invalid_argument("operand not defined");

  if (arg1.shape() != arg2.shape())
    throw invalid_argument("operating with incompatible shapes");

  dest.shape_ = arg1.shape();

  if (arg1.example_val() && arg2.example_val())
  {
    PackedVal dest_example_val;
    clear_data_evaluator_.operate_binary(op_code, *arg1.example_val(), *arg2.example_val(), dest_example_val);
    dest.example_val_ = move(dest_example_val);
  }

  vector<Term *> operands{arg1_term, arg2_term};
  dest.id_ = insert_op_term(move(op_code), move(operands))->id();
}

template <typename T>
void Func::set_output(const T &output, string label)
{
  if (auto term = data_flow_.find_term(output.id()); term)
    data_flow_.set_output(term, ParamTermInfo{move(label), output.example_val()});
  else
    throw invalid_argument("object not defined");
}

Term *Func::insert_op_term(OpCode op_code, vector<Term *> operands, bool &inserted)
{
  if (Compiler::const_folding_enabled())
  {
    vector<PackedVal> const_vals;
    bool can_fold = true;
    for (auto operand : operands)
    {
      if (auto const_val = data_flow_.get_const_val(operand); const_val)
        const_vals.push_back(*const_val);
      else
      {
        can_fold = false;
        break;
      }
    }
    if (can_fold)
    {
      PackedVal dest_val;
      clear_data_evaluator_.operate(op_code, const_vals, dest_val);
      return data_flow_.insert_const(dest_val);
    }
  }

  return data_flow_.insert_op(move(op_code), move(operands), inserted);
}

void Func::replace_term_with(Term *term1, Term *term2)
{
  if (*term1 == *term2)
    return;

  data_flow_.replace(term1, term2);
  // if (auto output_info_node = outputs_info_.extract(term1->id()); !output_info_node.empty())
  // {
  //   const auto &output_info = output_info_node.mapped();
  //   if (output_info.example_val_)
  //     set_output(term2, move(output_info.label_));
  //   else
  //     set_output(term2, move(output_info.label_), *output_info.example_val_);
  //   data_flow_.unset_output(term1);
  // }
  // delete_term_cascade(term1);
}

// void Func::set_output(Term *term, string label)
// {
//   auto [it, inserted] = outputs_info_.emplace(term->id(), ParamTermInfo{move(label), nullopt});
//   if (inserted)
//     data_flow_.set_output(term);
//   else
//     clog << "the output term with label '" << label << "' was merged with the output term with label '"
//          << it->second.label_ << "', and will be accessible with the latter label\n";
// }

// void Func::set_output(Term *term, string label, PackedVal example_val)
// {
//   auto [it, inserted] = outputs_info_.emplace(term->id(), ParamTermInfo{move(label), move(example_val)});
//   if (inserted)
//     data_flow_.set_output(term);
//   else
//     clog << "the output term with label '" << label << "' was merged with the output term with label '"
//          << it->second.label_ << "', and will be accessible with the latter label\n";
// }

// void Func::remove_dead_code()
// {
//   for (auto leaf : data_flow_.prune_unreachabe_terms())
//     clean_deleted_leaf_term(leaf);
// }

// void Func::delete_term_cascade(Term *term)
// {
//   for (auto leaf : data_flow_.delete_term_cascade(term))
//     clean_deleted_leaf_term(leaf);
// }

// void Func::clean_deleted_leaf_term(size_t id)
// {
//   if (!inputs_info_.erase(id))
//   {
//     if (auto it = const_values_.find(id); it != const_values_.end())
//     {
//       if (Compiler::cse_enabled())
//         values_to_const_terms_.erase(it->second);
//       const_values_.erase(it);
//     }
//     else
//       throw logic_error("invalid leaf, non-input and non-const");
//   }
// }

// init_input
template void Func::init_input(Ciphertext &, string);
template void Func::init_input(Plaintext &, string);
// init_const
template void Func::init_const(Ciphertext &, PackedVal);
template void Func::init_const(Plaintext &, PackedVal);
// set_output
template void Func::set_output(const Ciphertext &, string);
template void Func::set_output(const Plaintext &, string);
// operate_unary
template void Func::operate_unary(OpCode, const Ciphertext &, Ciphertext &);
template void Func::operate_unary(OpCode, const Plaintext &, Ciphertext &);
template void Func::operate_unary(OpCode, const Plaintext &, Plaintext &);
// operate_binary
template void Func::operate_binary(OpCode, const Ciphertext &, const Ciphertext &, Ciphertext &);
template void Func::operate_binary(OpCode, const Ciphertext &, const Plaintext &, Ciphertext &);
template void Func::operate_binary(OpCode, const Plaintext &, const Ciphertext &, Ciphertext &);
template void Func::operate_binary(OpCode, const Plaintext &, const Plaintext &, Plaintext &);
} // namespace fheco::ir