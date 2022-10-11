#include<memory>
#include"program.hpp"
#include"term.hpp"
#include"ciphertext.hpp"

extern std::shared_ptr<ir::Program> program;

namespace fhecompiler
{

size_t Ciphertext::ciphertext_id=0;

using Ptr = std::shared_ptr<ir::Term>;

inline void set_new_label(Ciphertext& ct)
{
  ct.set_label(datatype::ct_label_prefix+std::to_string(Ciphertext::ciphertext_id++));
}

void compound_operate(Ciphertext& lhs, const Ciphertext& rhs, ir::OpCode opcode)
{

  auto lhs_node_ptr = program->insert_node<Ciphertext>(lhs);
  auto rhs_node_ptr = program->insert_node<Ciphertext>(rhs);

  std::string old_label = lhs.get_label();

  set_new_label(lhs);
  auto new_operation_node_ptr = program->insert_operation_node(opcode, {lhs_node_ptr, rhs_node_ptr}, lhs.get_label(), ir::ciphertextType);
  if(lhs_node_ptr->get_output_flag()) 
  {
    lhs_node_ptr->set_output_flag(false);
    new_operation_node_ptr->set_output_flag(true);
  }

  auto table_entry_opt = program->get_entry_form_constants_table(old_label);
  if(table_entry_opt != std::nullopt)
  {
    ir::ConstantTableEntry& table_entry = *table_entry_opt;
    if(std::get_if<std::string>(&(table_entry.get_entry_value())))
      program->insert_new_entry_from_existing_with_delete(lhs.get_label(), old_label);
  }

}

Ciphertext::Ciphertext(Plaintext& pt): label(datatype::ct_label_prefix+std::to_string(Ciphertext::ciphertext_id++))
{
  program->insert_node<Ciphertext>(*this);
  program->insert_entry_in_constants_table({this->label, {ir::ConstantTableEntry::constant, {pt.get_label()}}});
}

Ciphertext operate(Ciphertext& ct, ir::OpCode opcode, const std::vector<Ptr>& operands, ir::TermType term_type, bool is_output=false)
{
  set_new_label(ct);
  program->insert_operation_node(opcode, operands, ct.get_label(), term_type);
  return ct;
}

void Ciphertext::set_as_output() const
{

  auto this_node_ptr = program->find_node_in_data_flow(this->label);
  this_node_ptr->set_output_flag(true);
  auto constant_table_entry = program->get_entry_form_constants_table(this->label);
  if(constant_table_entry == std::nullopt)
  {
    program->insert_entry_in_constants_table({this->label, {ir::ConstantTableEntry::output, this->label+"_"+datatype::output_tag}});
  }
  else
  {
    ir::ConstantTableEntry& table_entry = *constant_table_entry;
    table_entry.set_entry_type(ir::ConstantTableEntry::output);
  }

}

Ciphertext::Ciphertext(std::string tag, bool output_flag, bool input_flag): label(datatype::ct_label_prefix+std::to_string(Ciphertext::ciphertext_id++))
{

  //we are expecting from the user to provide a tag for input
  auto node_ptr = program->insert_node<Ciphertext>(*this);
  node_ptr->set_iutput_flag(input_flag);
  node_ptr->set_output_flag(output_flag);
  ir::ConstantTableEntry::ConstantTableEntryType entry_type;

  if( input_flag && output_flag ) entry_type = ir::ConstantTableEntry::io;
  else if( input_flag ) entry_type = ir::ConstantTableEntry::input;
  else if( output_flag ) entry_type = ir::ConstantTableEntry::output;
  else entry_type = ir::ConstantTableEntry::constant;

  ir::ConstantTableEntry::EntryValue entry_value = tag;
  program->insert_entry_in_constants_table({this->label, {entry_type, entry_value}});

}

Ciphertext& Ciphertext::operator=(const Ciphertext& ct_copy)
{
  auto this_node_ptr = program->find_node_in_data_flow(this->get_label());

  if( this_node_ptr->get_output_flag() )
  {
    auto ct_copy_node_ptr = program->insert_node<Ciphertext>(ct_copy);
    //inserting new output in data flow as assignement, and in the constatns_table but this time we insert it as a symbol with tag
    std::string old_label = this->label;
    set_new_label(*this);
    program->insert_new_entry_from_existing_with_delete(this->label, old_label);
    auto new_assign_operation = program->insert_operation_node(ir::assign, {ct_copy_node_ptr}, this->label, ir::ciphertextType);
    new_assign_operation->set_output_flag(true);
  }

  else this->label = ct_copy.get_label();

  return *this;
}

Ciphertext::Ciphertext(const Ciphertext& ct_copy): label(datatype::ct_label_prefix + std::to_string(ciphertext_id++))
{

  auto ct_copy_node_ptr = program->insert_node<Ciphertext>(ct_copy);
  program->insert_operation_node(ir::assign, {ct_copy_node_ptr}, this->label, ir::ciphertextType);
  //std::cout << this->label << " = " << ct_copy.get_label() << "\n";

}

Ciphertext& Ciphertext::operator+=(const Ciphertext& rhs) 
{

  compound_operate(*this, rhs, ir::add);
  return *this;

}


Ciphertext& Ciphertext::operator*=(const Ciphertext& rhs) 
{
  compound_operate(*this, rhs, ir::mul);
  return *this;
}

Ciphertext& Ciphertext::operator-=(const Ciphertext& rhs) 
{
  compound_operate(*this, rhs, ir::sub);
  return *this;
}


Ciphertext operator+(Ciphertext& lhs, const Ciphertext& rhs)
{

  auto lhs_node_ptr = program->insert_node<Ciphertext>(lhs);
  auto rhs_node_ptr = program->insert_node<Ciphertext>(rhs);
  return operate(lhs, ir::add, {lhs_node_ptr, rhs_node_ptr}, ir::ciphertextType);

}

Ciphertext operator*(Ciphertext& lhs, const Ciphertext& rhs)
{

  auto lhs_node_ptr = program->insert_node<Ciphertext>(lhs);
  auto rhs_node_ptr = program->insert_node<Ciphertext>(rhs);
  return operate(lhs, ir::mul, {lhs_node_ptr, rhs_node_ptr}, ir::ciphertextType);

}

Ciphertext operator-(Ciphertext& lhs, const Ciphertext& rhs)
{

  auto lhs_node_ptr = program->insert_node<Ciphertext>(lhs);
  auto rhs_node_ptr = program->insert_node<Ciphertext>(rhs);
  return operate(lhs, ir::sub, {lhs_node_ptr, rhs_node_ptr}, ir::ciphertextType);

}

Ciphertext operator-(Ciphertext& rhs) 
{
  auto rhs_node_ptr = program->insert_node<Ciphertext>(rhs);
  return operate(rhs, ir::negate, {rhs_node_ptr}, ir::ciphertextType);
}

} //namespace fhecompiler
