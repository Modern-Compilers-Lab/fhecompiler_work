#pragma once

#include "op_code.hpp"
#include "term_type.hpp"
#include <cstddef>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fhecompiler
{
namespace ir
{
  class Term
  {
  public:
    Term(OpCode op_code, std::vector<Term *> operands)
      : id_{count_++}, op_code_{std::move(op_code)}, operands_{std::move(operands)},
        type_{OpCode::deduce_result_type(op_code_, operands_)}, parents_{}
    {}

    Term(TermType type) : id_{count_++}, op_code_{OpCode::nop}, operands_{}, type_{std::move(type)}, parents_{} {}

    inline std::size_t id() const { return id_; }

    inline const OpCode &op_code() const { return op_code_; }

    inline const std::vector<Term *> &operands() const { return operands_; }

    inline const TermType &type() const { return type_; }

    inline bool is_operation() const { return op_code_ != OpCode::nop; }

  private:
    // to construct temp object used as search keys for sets
    Term(std::size_t id) : id_{id}, op_code_{OpCode::nop}, operands_{}, type_{TermType::ciphertext}, parents_{} {}

    struct ParentKey
    {
      const OpCode *op_code;
      const std::vector<Term *> *operands;
    };

    struct HashParentKey
    {
      std::size_t operator()(const ParentKey &k) const;
    };

    struct EqualParentKey
    {
      bool operator()(const ParentKey &lhs, const ParentKey &rhs) const;
    };

    static std::size_t count_;

    void replace_in_parents_with(Term *t);

    std::size_t id_;

    OpCode op_code_;

    std::vector<Term *> operands_;

    TermType type_;

    // it seems we don't need explicit parent multiplicity per entry value
    std::unordered_map<ParentKey, Term *, HashParentKey, EqualParentKey> parents_;

    friend class DAG;
  };

  inline bool operator==(const Term &lhs, const Term &rhs)
  {
    return lhs.id() == rhs.id();
  }

  inline bool operator<(const Term &lhs, const Term &rhs)
  {
    return lhs.id() < rhs.id();
  }

  inline bool operator!=(const Term &lhs, const Term &rhs)
  {
    return !(lhs == rhs);
  }

  inline bool operator>(const Term &lhs, const Term &rhs)
  {
    return rhs < lhs;
  }

  inline bool operator<=(const Term &lhs, const Term &rhs)
  {
    return !(lhs > rhs);
  }

  inline bool operator>=(const Term &lhs, const Term &rhs)
  {
    return !(lhs < rhs);
  }
} // namespace ir
} // namespace fhecompiler

namespace std
{
template <>
struct hash<fhecompiler::ir::Term>
{
  inline size_t operator()(const fhecompiler::ir::Term &term) const { return hash<std::size_t>()(term.id()); }
};
} // namespace std
