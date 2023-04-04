#include "fhecompiler.hpp"
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
using namespace fhecompiler;

Ciphertext reduce_add_lin(const Ciphertext &x, size_t vector_size)
{
  Ciphertext result = x;
  Ciphertext rotated_cipher = x;
  for (int i = 1; i < vector_size; ++i)
  {
    rotated_cipher <<= 1;
    result += rotated_cipher;
  }
  return result;
}

void l2_distance(size_t vector_size, bool use_log_reduction)
{
  Ciphertext c1("c1", -5, 5);
  Ciphertext c2("c2", -5, 5);
  Ciphertext slot_wise_diff = square(c1 - c2);
  if (use_log_reduction)
    reduce_add(slot_wise_diff).set_output("result");
  else
    reduce_add_lin(slot_wise_diff, vector_size).set_output("result");
}

int main(int argc, char **argv)
{
  size_t vector_size = 1024;
  if (argc > 1)
    vector_size = stoull(argv[1]);

  bool use_log_reduction = true;
  if (argc > 2)
  {
    std::stringstream ss(argv[2]);
    if (!(ss >> boolalpha >> use_log_reduction))
      throw invalid_argument("could not parse use_log_reduction to bools");
  }

  int trs_passes = 1;
  if (argc > 3)
    trs_passes = stoi(argv[3]);

  bool optimize = trs_passes > 0;

  cout << "vector_size: " << vector_size << ", "
       << "trs_passes: " << trs_passes << '\n';

  string func_name = "l2_distance";
  Compiler::create_func(func_name, vector_size, 16, true, Scheme::bfv);
  l2_distance(vector_size, use_log_reduction);
  ofstream init_ir_os(func_name + "_init_ir.dot");
  Compiler::draw_ir(init_ir_os);
  const auto &rand_inputs = Compiler::get_example_input_values();
  ofstream gen_code_os("he/gen_he_" + func_name + ".hpp");
  if (optimize)
    Compiler::compile(gen_code_os, trs_passes);
  else
    Compiler::compile_noopt(gen_code_os);
  ofstream final_ir_os(func_name + "_final_ir.dot");
  Compiler::draw_ir(final_ir_os);
  gen_code_os.close();
  auto outputs = Compiler::evaluate_on_clear(rand_inputs);
  if (outputs != Compiler::get_example_output_values())
    throw logic_error("compilation correctness-test failed");

  ofstream rand_example_os(func_name + "_rand_example.txt");
  Compiler::print_inputs_outputs(rand_example_os);
  return 0;
}
