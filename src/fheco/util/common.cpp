#include "fheco/util/common.hpp"
#include <stdexcept>
#include <variant>

using namespace std;

namespace fheco
{
namespace util
{
  void init_random(PackedVal &packed_val, integer slot_min, integer slot_max)
  {
    random_device rd;
    mt19937 rng(rd());
    uniform_int_distribution<integer> uni(slot_min, slot_max);
    for (auto it = packed_val.begin(); it != packed_val.end(); ++it)
      *it = uni(rng);
  }

  void print_io_terms_values(const ir::Function &func, ostream &os)
  {
    ios_base::fmtflags f(os.flags());
    os << boolalpha;

    os << func.clear_data_evaluator().slot_count() << " " << func.inputs_info().size() << " "
       << func.outputs_info().size() << '\n';
    for (const auto &in : func.inputs_info())
    {
      if (auto in_info_it = func.inputs_info().find(in.first); in_info_it == func.inputs_info().end())
        throw invalid_argument("no input with id was found");

      auto in_term = func.data_flow().find_term(in.first);
      os << in.second.label << " " << (in_term->type() == ir::TermType::ciphertext) << " "
         << (func.clear_data_evaluator().signedness() || func.clear_data_evaluator().delayed_reduction());
      if (in.second.example_val)
        os << " " << *in.second.example_val;
      os << '\n';
    }
    for (const auto &out : func.outputs_info())
    {
      if (auto out_info_it = func.outputs_info().find(out.first); out_info_it == func.outputs_info().end())
        throw invalid_argument("no output with id was found");

      auto out_term = func.data_flow().find_term(out.first);
      os << out.second.label << " " << (out_term->type() == ir::TermType::ciphertext) << " "
         << (func.clear_data_evaluator().signedness() || func.clear_data_evaluator().delayed_reduction());
      if (out.second.example_val)
        os << " " << *out.second.example_val;
      os << '\n';
    }
    os.flags(f);
  }

  void print_io_terms_values(
    const ir::Function &func, const ir::IOTermsInfo &inputs, const ir::IOTermsInfo &outputs, ostream &os)
  {
    ios_base::fmtflags f(os.flags());
    os << boolalpha;

    os << func.clear_data_evaluator().slot_count() << " " << inputs.size() << " " << outputs.size() << '\n';
    for (const auto &in : inputs)
    {
      auto in_term = func.data_flow().find_term(in.first);
      if (!in_term)
        throw invalid_argument("term with id not found");

      os << in.second.label << " " << (in_term->type() == ir::TermType::ciphertext) << " "
         << (func.clear_data_evaluator().signedness() || func.clear_data_evaluator().delayed_reduction());
      if (in.second.example_val)
        os << " " << *in.second.example_val;
      os << '\n';
    }
    for (const auto &out : outputs)
    {
      auto out_term = func.data_flow().find_term(out.first);
      if (!out_term)
        throw invalid_argument("term with id not found");

      os << out.second.label << " " << (out_term->type() == ir::TermType::ciphertext) << " "
         << (func.clear_data_evaluator().signedness() || func.clear_data_evaluator().delayed_reduction());
      if (out.second.example_val)
        os << " " << *out.second.example_val;
      os << '\n';
    }
    os.flags(f);
  }

  void print_io_terms_values(const ir::IOTermsInfo &io_terms_values, size_t lead_trail_size, ostream &os)
  {
    for (const auto &term : io_terms_values)
    {
      os << term.first << " " << term.second.label;
      if (term.second.example_val)
      {
        os << " ";
        print_packed_val(*term.second.example_val, lead_trail_size, os);
      }
      os << '\n';
    }
  }

  void print_terms_values(const ir::TermsValues &terms_values, size_t lead_trail_size, ostream &os)
  {
    for (const auto &term : terms_values)
    {
      os << term.first << " ";
      visit(
        ir::overloaded{
          [lead_trail_size, &os](const PackedVal &val) { print_packed_val(val, lead_trail_size, os); },
          [&os](ScalarVal val) {
            os << val;
          }},
        term.second);
      os << '\n';
    }
  }

  void print_packed_val(const PackedVal &packed_val, size_t lead_trail_size, ostream &os)
  {
    size_t slot_count = packed_val.size();
    if (slot_count < 2 * lead_trail_size)
      throw invalid_argument("packed_val size must at least twice lead_trail_size");

    if (slot_count == 0)
      return;

    for (size_t i = 0; i < lead_trail_size; ++i)
      os << packed_val[i] << " ";
    if (slot_count > 2 * lead_trail_size)
      os << "... ";
    for (size_t i = slot_count - lead_trail_size; i < slot_count - 1; ++i)
      os << packed_val[i] << " ";
    os << packed_val.back();
  }
} // namespace util
} // namespace fheco

namespace std
{
ostream &operator<<(ostream &os, const fheco::ir::IOTermsInfo &io_terms_values)
{
  for (const auto &term : io_terms_values)
  {
    os << term.first << " " << term.second.label;
    if (term.second.example_val)
      os << " " << *term.second.example_val;
    os << '\n';
  }
  return os;
}

ostream &operator<<(ostream &os, const fheco::ir::TermsValues &terms_values)
{
  for (const auto &term : terms_values)
  {
    os << term.first << " ";
    visit(
      fheco::ir::overloaded{
        [&os](const fheco::PackedVal &val) { os << val; },
        [&os](fheco::ScalarVal val) {
          os << val;
        }},
      term.second);
    os << '\n';
  }
  return os;
}

ostream &operator<<(ostream &os, const fheco::PackedVal &packed_val)
{
  if (packed_val.size() == 0)
    return os;

  for (size_t i = 0; i < packed_val.size() - 1; ++i)
    os << packed_val[i] << " ";
  os << packed_val.back();
  return os;
}
} // namespace std
