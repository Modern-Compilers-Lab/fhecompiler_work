#pragma once

#include "dag.hpp"
#include "fhecompiler_const.hpp"
#include "ir_const.hpp"
#include "term.hpp"
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>

namespace ir
{

struct ConstantTableEntry
{

  struct Encrypt
  {
    std::string plaintext_label;
    // to be extended in the future maybe
  };

  using ScalarValue = std::variant<int64_t, double, uint64_t>;

  using VectorValue = std::variant<std::vector<int64_t>, std::vector<double>>;

  using ConstantValue = std::variant<ScalarValue, VectorValue, Encrypt>;

  struct EntryValue
  {

    std::optional<ConstantValue> value;
    std::string tag;

    EntryValue(const std::string &_tag, std::optional<ConstantValue> _value) : value(_value), tag(_tag) {}

    EntryValue(const std::string _tag) : tag(_tag), value(std::nullopt) {}

    EntryValue(std::optional<ConstantValue> _value) : value(_value), tag("") {}

    std::string get_tag() { return this->tag; }

    ~EntryValue() = default;
  };

  ConstantTableEntryType entry_type;
  EntryValue entry_value;

public:
  ConstantTableEntry() = default;

  ConstantTableEntry(const ConstantTableEntry &entry_copy) = default;
  ConstantTableEntry &operator=(const ConstantTableEntry &entry_copy) = default;

  ConstantTableEntry(ConstantTableEntry &&entry_copy) = default;
  ConstantTableEntry &operator=(ConstantTableEntry &entry_copy) = default;

  ConstantTableEntry(ConstantTableEntryType _type, EntryValue _value) : entry_type(_type), entry_value(_value) {}

  void set_entry_type(ConstantTableEntryType _type) { entry_type = _type; }

  void set_entry_value(const EntryValue &_value) { entry_value = _value; }

  void set_entry_value(const ConstantValue &_value) { entry_value.value = _value; }

  void set_entry_tag(const std::string &_tag)
  {
    if (_tag.length())
      entry_value.tag = _tag;
  }

  EntryValue &get_entry_value() { return entry_value; }

  ConstantTableEntryType get_entry_type() const { return this->entry_type; }
};

class Program
{

private:
  std::string program_tag; // program_tag defines the name of the main evaluation function which will be generated

  std::unique_ptr<DAG> data_flow; // data_flow points to the IR which is a Directed Acyclic Graph (DAG)

  std::unordered_map<std::string, ConstantTableEntry>
    constants_table; // we will have a symbol table, the data structure is a hash table

  fhecompiler::Scheme program_scheme;

  fhecompiler::Backend target_backed;

  double scale = 0.0; // for ckks

  size_t number_of_slots;

  size_t plain_modulus;

  // size_t dimension;

public:
  using Ptr = std::shared_ptr<Term>;

  Program() = delete;

  Program(const std::string &tag_value) : program_tag{tag_value} { data_flow = std::make_unique<DAG>(); }

  ~Program() {}

  void set_targeted_backed(fhecompiler::Backend backend) { target_backed = backend; }

  fhecompiler::Backend get_targeted_backend() { return this->target_backed; }

  Ptr insert_operation_node_in_dataflow(
    OpCode _opcode, const std::vector<Ptr> &_operands, std::string label, TermType term_type);

  Ptr find_node_in_dataflow(const std::string &label) const;

  void set_symbol_as_output(const std::string &label, const std::string &tag);

  void delete_node_from_dataflow(const std::string &node_label);

  template <typename T>
  Ptr insert_node_in_dataflow(const T &operand)
  {
    auto node_ptr_in_program = find_node_in_dataflow(operand.get_label());
    if (node_ptr_in_program)
    {
      return node_ptr_in_program;
    }
    Ptr new_term = std::make_shared<Term>(operand);
    this->data_flow->insert_node(new_term, this->type_of(new_term->get_label()) == ConstantTableEntryType::output);
    return new_term;
  }

  void set_node_operands(const std::string &node_label, const std::vector<Ptr> &new_opreands);

  void set_scheme(fhecompiler::Scheme program_scheme_value) { program_scheme = program_scheme_value; }

  void insert_entry_in_constants_table(std::pair<std::string, ConstantTableEntry> table_entry);

  bool delete_entry_from_constants_table(std::string entry_key);

  void delete_node_from_outputs(const std::string &key);

  bool insert_new_entry_from_existing(std::string new_entry_key, std::string exsisting_entry_key);

  bool insert_new_entry_from_existing_with_delete(std::string new_entry_key, std::string exsisting_entry_key);

  ConstantTableEntryType type_of(const std::string &label);

  void set_plain_modulus(size_t modulus) { this->plain_modulus = modulus; }

  size_t get_plain_modulus() const { return this->plain_modulus; }

  // size_t get_dimension() const { return this->dimension; }

  void sort_dataflow();

  const std::vector<Ptr> &get_dataflow_sorted_nodes(bool clear_existing_order = false) const;

  const std::unordered_map<std::string, Ptr> &get_outputs_nodes() const;

  std::optional<std::reference_wrapper<ConstantTableEntry>> get_entry_form_constants_table(const std::string &);

  const std::string &get_program_tag() const { return this->program_tag; }

  fhecompiler::Scheme get_encryption_scheme() const { return this->program_scheme; }

  void set_scale(double _scale) { scale = _scale; }

  double get_scale() { return scale; }

  void set_number_of_slots(size_t nb_slots) { number_of_slots = nb_slots; }
  size_t get_number_of_slots() { return number_of_slots; }
};

} // namespace ir
