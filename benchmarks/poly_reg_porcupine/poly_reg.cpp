#include "fheco/fheco.hpp"
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace std;
using namespace fheco;

void poly_reg_baseline()
{
  Ciphertext c0("c0");
  Ciphertext c1("c1");
  Ciphertext c2("c2");
  Ciphertext c3("c3");
  Ciphertext c4("c4");
  Ciphertext c_result = c1 - (square(c0) * c4 + c0 * c3 + c2);
  c_result.set_output("c_result");
}

void poly_reg_synthesized()
{
  Ciphertext c0("c0");
  Ciphertext c1("c1");
  Ciphertext c2("c2");
  Ciphertext c3("c3");
  Ciphertext c4("c4");
  Ciphertext c5 = c0 * c4;
  Ciphertext c6 = c5 + c3;
  Ciphertext c7 = c0 * c6;
  Ciphertext c8 = c7 + c2;
  Ciphertext c_result = c1 - c8;
  c_result.set_output("c_result");
}

void print_bool_arg(bool arg, const string &name, ostream &os)
{
  os << (arg ? name : "no_" + name);
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

  int64_t max_iter = 1000;
  if (argc > 4)
    max_iter = stoull(argv[4]);

  bool rewrite_created_sub_terms = true;
  if (argc > 5)
    rewrite_created_sub_terms = stoi(argv[5]);

  bool cse = true;
  if (argc > 6)
    cse = stoi(argv[6]);

  bool cse_order_operands = true;
  if (argc > 7)
    cse_order_operands = stoi(argv[7]);

  bool const_folding = true;
  if (argc > 8)
    const_folding = stoi(argv[8]);

  print_bool_arg(call_quantifier, "call_quantifier", clog);
  clog << " ";
  clog << ruleset << " " << rewrite_heuristic << " " << max_iter << " ";
  print_bool_arg(rewrite_created_sub_terms, "rewrite_created_sub_terms", clog);
  clog << " ";
  print_bool_arg(cse, "cse", clog);
  clog << " ";
  print_bool_arg(cse_order_operands, "cse_order_operands", clog);
  clog << " ";
  print_bool_arg(const_folding, "const_folding", clog);
  clog << '\n';

  if (cse)
    Compiler::enable_cse();
  else
    Compiler::disable_cse();

  if (cse_order_operands)
    Compiler::enable_order_operands();
  else
    Compiler::disable_order_operands();

  if (const_folding)
    Compiler::enable_const_folding();
  else
    Compiler::disable_const_folding();

  chrono::high_resolution_clock::time_point time_start, time_end;
  chrono::duration<double, milli> time_sum(0);
  time_start = chrono::high_resolution_clock::now();
  string func_name = "poly_reg";
  Compiler::create_func(func_name, 1024, 20, true, false);

  poly_reg_baseline();

  string gen_name = "gen_he_" + func_name;
  string gen_path = "he/" + gen_name;
  ofstream header_os(gen_path + ".hpp");
  ofstream source_os(gen_path + ".cpp");

  Compiler::compile(
    ruleset, rewrite_heuristic, max_iter, rewrite_created_sub_terms, header_os, gen_name + ".hpp", source_os);

  time_end = chrono::high_resolution_clock::now();
  time_sum = time_end - time_start;
  cout << time_sum.count() << '\n';

  if (call_quantifier)
  {
    util::Quantifier quantifier1(Compiler::active_func());
    quantifier1.run_all_analysis();
    cout << quantifier1.he_depth_summary().max_xdepth_ << " " << quantifier1.he_depth_summary().max_depth_ << " ";
    cout << quantifier1.circuit_static_cost() << '\n';
  }

  return 0;
}
