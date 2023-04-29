#pragma once

#include "fheco/dsl/common.hpp"
#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>

namespace fheco::ir
{
using ConstVal = std::variant<PackedVal, ScalarVal>;

// overload pattern
template <class... Ts>
struct overloaded : Ts...
{
  using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

template <class>
inline constexpr bool always_false_v = false;

// https://github.com/HowardHinnant/hash_append/issues/7#issuecomment-629414712
template <typename T>
inline void hash_combine(std::size_t &seed, const T &val)
{
  seed ^= std::hash<T>{}(val) + 0x9e3779b97f4a7c15ULL + (seed << 12) + (seed >> 4);
}

struct HashConstVal
{
  size_t slot_count_;

  HashConstVal(size_t slot_count) : slot_count_{slot_count} {}

  std::size_t operator()(const ConstVal &value_var) const;
};

// order of definition is important for type deduction (OpCode::deduce_result_type)
enum class TermType
{
  ciphertext,
  plaintext,
  scalar
};

std::string term_type_str_repr(TermType);

enum class TermQualif
{
  temp,
  in,
  in_out,
  const_,
  out
};

struct ParamTermInfo
{
  std::string label_;
  std::optional<PackedVal> example_val_;
};

inline bool operator==(const ParamTermInfo &lhs, const ParamTermInfo &rhs)
{
  return lhs.label_ == rhs.label_ && lhs.example_val_ == rhs.example_val_;
}

inline bool operator!=(const ParamTermInfo &lhs, const ParamTermInfo &rhs)
{
  return !(lhs == rhs);
}

using IOTermsInfo = std::unordered_map<std::size_t, ParamTermInfo>;
using TermsValues = std::unordered_map<std::size_t, ConstVal>;
} // namespace fheco::ir
