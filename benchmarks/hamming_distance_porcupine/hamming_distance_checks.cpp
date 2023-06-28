#include "fheco/fheco.hpp"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace std;
using namespace fheco;

void hamming_distance_baseline()
{
  Plaintext d0(2);
  Ciphertext c0("c0", 0, 1);
  Ciphertext c1("c1", 0, 1);
  Ciphertext c2 = c1 + c0;
  Ciphertext c3 = -c2 * c1 - c2 * c0 + c2 * d0;
  Ciphertext c3_rot_2 = c3 << 2;
  Ciphertext c4 = c3_rot_2 + c3;
  Ciphertext c4_rot_1 = c4 << 1;
  Ciphertext c_result = c4_rot_1 + c4;

  c_result.set_output("c_result");
}

void hamming_distance_synthesized()
{
  Plaintext d0(2);
  Ciphertext c0("c0", 0, 1);
  Ciphertext c1("c1", 0, 1);
  Ciphertext c2 = c1 + c0;
  Ciphertext c2_neg = -c2;
  Ciphertext c3 = c2_neg + d0;
  Ciphertext c4 = c3 * c2;
  Ciphertext c4_rot_2 = c4 << 2;
  Ciphertext c5 = c4_rot_2 + c4;
  Ciphertext c5_rot_1 = c5 << 1;
  Ciphertext c_result = c5_rot_1 + c5;
  c_result.set_output("c_result");
}

void print_bool_arg(bool arg, const string &name, ostream &os)
{
  os << (arg ? name : "non_" + name);
}

int main(int argc, char **argv)
{
  bool call_quantifier = false;
  if (argc > 1)
    call_quantifier = stoi(argv[1]);

  auto ruleset = Compiler::Ruleset::joined;
  if (argc > 2)
    ruleset = static_cast<Compiler::Ruleset>(stoi(argv[2]));

  auto rewrite_heuristic = trs::RewriteHeuristic::bottom_up;
  if (argc > 3)
    rewrite_heuristic = static_cast<trs::RewriteHeuristic>(stoi(argv[3]));

  bool cse = true;
  if (argc > 4)
    cse = stoi(argv[4]);

  bool const_folding = true;
  if (argc > 5)
    const_folding = stoi(argv[5]);

  print_bool_arg(call_quantifier, "quantificateur", clog);
  clog << " ";
  clog << "trs_" << ruleset << " " << rewrite_heuristic;
  clog << " ";
  print_bool_arg(cse, "cse", clog);
  clog << " ";
  print_bool_arg(const_folding, "élimination_calculs_constants", clog);
  clog << '\n';

  string app_name = "hamming_distance";
  size_t slot_count = 4;
  int bit_width = 20;
  bool signdness = true;
  bool need_cyclic_rotation = false;

  clog << "\nfonction non optimisée\n";

  string noopt_func_name = app_name + "_noopt";
  const auto &noopt_func =
    Compiler::create_func(noopt_func_name, slot_count, bit_width, signdness, need_cyclic_rotation);

  hamming_distance_baseline();

  string noopt_gen_name = "gen_he_" + noopt_func_name;
  string noopt_gen_path = "he/" + noopt_gen_name;
  ofstream noopt_header_os(noopt_gen_path + ".hpp");
  ofstream noopt_source_os(noopt_gen_path + ".cpp");
  Compiler::gen_he_code(noopt_func, noopt_header_os, noopt_gen_name + ".hpp", noopt_source_os);

  util::Quantifier noopt_quantifier(noopt_func);
  if (call_quantifier)
  {
    cout << "\ncaractéristiques du circuit initial\n";
    noopt_quantifier.run_all_analysis();
    noopt_quantifier.print_info(cout);
    cout << endl;
  }

  ofstream noopt_ir_os(noopt_func_name + "_ir.dot");
  util::draw_ir(noopt_func, noopt_ir_os);

  clog << "\nfonction optimisée\n";

  if (cse)
  {
    Compiler::enable_cse();
    Compiler::enable_order_operands();
  }
  else
  {
    Compiler::disable_cse();
    Compiler::disable_order_operands();
  }

  if (const_folding)
    Compiler::enable_const_folding();
  else
    Compiler::disable_const_folding();

  string opt_func_name = app_name + "_opt";
  const auto &opt_func = Compiler::create_func(opt_func_name, slot_count, bit_width, signdness, need_cyclic_rotation);

  hamming_distance_baseline();

  string opt_gen_name = "gen_he_" + opt_func_name;
  string opt_gen_path = "he/" + opt_gen_name;
  ofstream opt_header_os(opt_gen_path + ".hpp");
  ofstream opt_source_os(opt_gen_path + ".cpp");

  Compiler::compile(opt_func, ruleset, rewrite_heuristic, opt_header_os, opt_gen_name + ".hpp", opt_source_os);

  auto noopt_obtained_outputs = util::evaluate_on_clear(noopt_func, opt_func->get_inputs_example_values());
  auto opt_obtained_outputs = util::evaluate_on_clear(opt_func, noopt_func->get_inputs_example_values());
  if (
    noopt_obtained_outputs != opt_func->get_outputs_example_values() ||
    opt_obtained_outputs != noopt_func->get_outputs_example_values())
    throw logic_error("compilation correctness-test failed");

  ofstream io_example_os(app_name + "_io_example.txt");
  util::print_io_terms_values(opt_func, io_example_os);

  ofstream opt_ir_os(opt_func_name + "_ir.dot");
  util::draw_ir(opt_func, opt_ir_os);

  if (call_quantifier)
  {
    cout << "\ncaractéristiques du circuit final\n";
    util::Quantifier opt_quantifier(opt_func);
    opt_quantifier.run_all_analysis();
    opt_quantifier.print_info(cout);

    cout << "\ntaux d'amélioration\n";
    auto diff_quantifier = (noopt_quantifier - opt_quantifier) / noopt_quantifier * 100;
    diff_quantifier.print_info(cout);
  }

  return 0;
}
