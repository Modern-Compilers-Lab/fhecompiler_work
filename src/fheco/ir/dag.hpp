#pragma once

#include "fheco/ir/common.hpp"
#include "fheco/ir/op_code.hpp"
#include "fheco/ir/term.hpp"
#include <cstddef>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fheco
{
namespace ir
{
  class DAG
  {
  public:
    struct HashTermPtr
    {
      inline std::size_t operator()(const Term *p) const { return std::hash<Term>()(*p); }
    };

    struct EqualTermPtr
    {
      inline bool operator()(const Term *lhs, const Term *rhs) const { return *lhs == *rhs; }
    };

    struct CompareTermPtr
    {
      inline bool operator()(const Term *lhs, const Term *rhs) const { return *lhs < *rhs; }
    };

    using TermPtrSet = std::unordered_set<Term *, HashTermPtr, EqualTermPtr>;

    DAG() : outputs_{}, sorted_terms_{}, valid_top_sort_{true}, terms_{} {}

    ~DAG();

    DAG(const DAG &other) = delete;

    DAG(DAG &&other);

    DAG &operator=(const DAG &other) = delete;

    DAG &operator=(DAG &&other);

    Term *insert_operation_term(OpCode op_code, std::vector<Term *> operands);

    Term *insert_leaf(TermType type);

    Term *find_term(std::size_t id) const;

    Term *find_term(const OpCode &op_code, const std::vector<Term *> &operands) const;

    void prune_unreachable_terms();

    void replace_term_with(Term *term1, Term *term2);

    void set_output(Term *term);

    void unset_output(Term *term);

    inline const std::vector<const Term *> &get_top_sorted_terms()
    {
      if (valid_top_sort_)
        return sorted_terms_;

      topological_sort();
      return sorted_terms_;
    }

    const TermPtrSet &output_terms() const { return outputs_; }

  private:
    inline bool is_output(Term *term) const { return outputs_.find(t) != outputs_.end(); }

    void delete_non_output_source_cascade(Term *term);

    void topological_sort();

    TermPtrSet outputs_;

    std::vector<const Term *> sorted_terms_;

    bool valid_top_sort_;

    TermPtrSet terms_;
  };
} // namespace ir
} // namespace fheco
