#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

namespace ir
{
template <class... Ts>
struct overloaded : Ts...
{
  using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

using ScalarValue = std::variant<std::uint64_t, std::int64_t>;

using VectorValue = std::variant<std::vector<std::uint64_t>, std::vector<std::int64_t>>;

using ConstantValue = std::variant<VectorValue, ScalarValue>;

enum class TermType
{
  undefined,
  rawData,
  scalar,
  ciphertext,
  plaintext
};

enum class OpCode
{
  undefined,
  assign,
  encrypt,
  add,
  add_plain,
  mul,
  mul_plain,
  sub,
  sub_plain,
  rotate,
  rotate_rows,
  rotate_columns,
  square,
  negate,
  exponentiate,
  modswitch,
  relinearize,
  rescale,
  encode,
  decode
};

const std::unordered_set<OpCode> non_commutative_ops = {OpCode::sub};

enum class ConstantTableEntryType
{
  undefined,
  constant,
  input,
  output,
  temp
};

inline std::unordered_map<ir::TermType, std::string> term_type_label_map = {
  {ir::TermType::scalar, "scalar"}, {ir::TermType::plaintext, "plaintext"}, {ir::TermType::ciphertext, "ciphertext"}};

inline std::unordered_map<ir::OpCode, std::string> str_opcode = {
  {OpCode::undefined, "undefined"},
  {OpCode::add, "add"},
  {OpCode::add_plain, "add_plain"},
  {OpCode::mul, "mul"},
  {OpCode::mul_plain, "mul_plain"},
  {OpCode::sub, "sub"},
  {OpCode::sub_plain, "sub_plain"},
  {OpCode::negate, "negate"},
  {OpCode::modswitch, "modswitch"},
  {OpCode::relinearize, "relinearize"},
  {OpCode::exponentiate, "exponentiate"},
  {OpCode::rescale, "rescale"},
  {OpCode::rotate, "rotate"},
  {OpCode::rotate_columns, "rotate_columns"},
  {OpCode::rotate_rows, "rotate_rows"},
  {OpCode::decode, "decode"},
  {OpCode::encode, "encode"}};

/*
  Encode, Decode
*/
} // namespace ir
