#include "fheco/fheco.hpp"
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace std;
using namespace fheco;

void box_blur_baseline()
{
  Ciphertext c0("c0");
  Ciphertext c1 = c0 << 1;
  Ciphertext c2 = c0 << 5;
  Ciphertext c3 = c0 << 6;
  Ciphertext c4 = c1 + c0;
  Ciphertext c5 = c2 + c3;
  Ciphertext c6 = c4 + c5;
  c6.set_output("c6");
}

void box_blur_synthesized()
{
  Ciphertext c0("c0");
  Ciphertext c1 = c0 << 1;
  Ciphertext c2 = c0 + c1;
  Ciphertext c3 = c2 << 5;
  Ciphertext c4 = c2 + c3;
  c4.set_output("c4");
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

  bool cse = true;
  if (argc > 4)
    cse = stoi(argv[4]);

  bool cse_order_operands = true;
  if (argc > 5)
    cse_order_operands = stoi(argv[5]);

  bool const_folding = true;
  if (argc > 6)
    const_folding = stoi(argv[6]);

  print_bool_arg(call_quantifier, "call_quantifier", clog);
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
  string func_name = "box_blur";
  // Compiler::create_func(func_name, 13, 20, false, true);
  Compiler::create_func(func_name, 16, 20, false, true);

  box_blur_baseline();

  string gen_name = "gen_he_" + func_name;
  string gen_path = "he/" + gen_name;
  ofstream header_os(gen_path + ".hpp");
  ofstream source_os(gen_path + ".cpp");

  Compiler::compile(ruleset, rewrite_heuristic, header_os, gen_name + ".hpp", source_os);

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
