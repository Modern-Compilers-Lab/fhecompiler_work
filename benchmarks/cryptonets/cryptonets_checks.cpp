#include "fheco/fheco.hpp"
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "ml.hpp"

using namespace std;
using namespace fheco;

void cryptonets(
  const vector<size_t> &x_shape, const vector<size_t> &w1_shape, const vector<size_t> &b1_shape,
  const vector<size_t> &w4_shape, const vector<size_t> &b4_shape, const vector<size_t> &w8_shape,
  const vector<size_t> &b8_shape)
{
  // declare inputs
  int x_min_val = -10;
  int x_max_val = 10;
  int wb_min = -10;
  int wb_max = 10;
  vector<vector<vector<Ciphertext>>> x;
  for (size_t i = 0; i < x_shape[0]; ++i)
  {
    x.push_back(vector<vector<Ciphertext>>());
    for (size_t j = 0; j < x_shape[1]; ++j)
    {
      x[i].push_back(vector<Ciphertext>());
      for (size_t k = 0; k < x_shape[2]; ++k)
        x[i][j].push_back(
          Ciphertext("x[" + to_string(i) + "][" + to_string(j) + "][" + to_string(k) + "]", x_min_val, x_max_val));
    }
  }

  vector<vector<vector<vector<Plaintext>>>> w1;
  for (size_t i = 0; i < w1_shape[0]; ++i)
  {
    w1.push_back(vector<vector<vector<Plaintext>>>());
    for (size_t j = 0; j < w1_shape[1]; ++j)
    {
      w1[i].push_back(vector<vector<Plaintext>>());
      for (size_t k = 0; k < w1_shape[2]; ++k)
      {
        w1[i][j].push_back(vector<Plaintext>());
        for (size_t l = 0; l < w1_shape[3]; ++l)
          w1[i][j][k].push_back(Plaintext(
            "w1[" + to_string(i) + "][" + to_string(j) + "][" + to_string(k) + "][" + to_string(l) + "]", wb_min,
            wb_max));
      }
    }
  }

  vector<Plaintext> b1;
  for (size_t i = 0; i < b1_shape[0]; ++i)
    b1.push_back(Plaintext("b1[" + to_string(i) + "]", wb_min, wb_max));

  vector<vector<vector<vector<Plaintext>>>> w4;
  for (size_t i = 0; i < w4_shape[0]; ++i)
  {
    w4.push_back(vector<vector<vector<Plaintext>>>());
    for (size_t j = 0; j < w4_shape[1]; ++j)
    {
      w4[i].push_back(vector<vector<Plaintext>>());
      for (size_t k = 0; k < w4_shape[2]; ++k)
      {
        w4[i][j].push_back(vector<Plaintext>());
        for (size_t l = 0; l < w4_shape[3]; ++l)
          w4[i][j][k].push_back(Plaintext(
            "w4[" + to_string(i) + "][" + to_string(j) + "][" + to_string(k) + "][" + to_string(l) + "]", wb_min,
            wb_max));
      }
    }
  }

  vector<Plaintext> b4;
  for (size_t i = 0; i < b4_shape[0]; ++i)
    b4.push_back(Plaintext("b4[" + to_string(i) + "]", wb_min, wb_max));

  vector<vector<Plaintext>> w8;
  for (size_t i = 0; i < w8_shape[0]; ++i)
  {
    w8.push_back(vector<Plaintext>());
    for (size_t j = 0; j < w8_shape[1]; ++j)
      w8[i].push_back(Plaintext("w8[" + to_string(i) + "][" + to_string(j) + "]", wb_min, wb_max));
  }

  vector<Plaintext> b8;
  for (size_t i = 0; i < b8_shape[0]; ++i)
    b8.push_back(Plaintext("b8[" + to_string(i) + "]", wb_min, wb_max));

  // show passed shapes
  cout << "x: ";
  print_vector(shape(x), cout);
  cout << '\n';
  cout << "w1: ";
  print_vector(shape(w1), cout);
  cout << '\n';
  cout << "b1: ";
  print_vector(shape(b1), cout);
  cout << '\n';
  cout << "w4: ";
  print_vector(shape(w4), cout);
  cout << '\n';
  cout << "b4: ";
  print_vector(shape(b4), cout);
  cout << '\n';
  cout << "w8: ";
  print_vector(shape(w8), cout);
  cout << '\n';
  cout << "b8: ";
  print_vector(shape(b8), cout);
  cout << '\n';
  // predict
  auto y = predict(x, w1, b1, w4, b4, w8, b8);
  // declare outputs
  for (size_t i = 0; i < y.size(); ++i)
    y[i].set_output("y[" + to_string(i) + "]");
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
  print_bool_arg(const_folding, "const_folding", clog);
  clog << '\n';

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

  string func_name = "cryptonets";
  const auto &func = Compiler::create_func(func_name, 8192, 17, true, false);

  cryptonets(
    vector<size_t>{28, 28, 1}, vector<size_t>{5, 5, 1, 5}, vector<size_t>{5}, vector<size_t>{5, 5, 5, 10},
    vector<size_t>{10}, vector<size_t>{40, 10}, vector<size_t>{10});

  util::Quantifier init_quantifier(func);
  if (call_quantifier)
  {
    cout << "\ncaractéristiques du circuit initial\n";
    init_quantifier.run_all_analysis();
    init_quantifier.print_info(cout);
  }

  ofstream init_ir_os(func_name + "_init_ir.dot");
  util::draw_ir(func, init_ir_os);

  const auto &rand_inputs = func->data_flow().inputs_info();

  string gen_name = "gen_he_" + func_name;
  string gen_path = "he/" + gen_name;
  ofstream header_os(gen_path + ".hpp");
  ofstream source_os(gen_path + ".cpp");

  Compiler::compile(ruleset, rewrite_heuristic, header_os, gen_name + ".hpp", source_os);

  auto outputs = util::evaluate_on_clear(func, rand_inputs);
  if (outputs != func->data_flow().outputs_info())
    throw logic_error("compilation correctness-test failed");

  ofstream rand_example_os(func_name + "_rand_example.txt");
  util::print_io_terms_values(func, rand_example_os);

  ofstream final_ir_os(func_name + "_final_ir.dot");
  util::draw_ir(func, final_ir_os);

  if (call_quantifier)
  {
    cout << "\ncaractéristiques du circuit final\n";
    util::Quantifier final_quantifier(func);
    final_quantifier.run_all_analysis();
    final_quantifier.print_info(cout);

    cout << "\ntaux d'amélioration\n";
    auto diff_quantifier = (init_quantifier - final_quantifier) / init_quantifier * 100;
    diff_quantifier.print_info(cout);
  }

  return 0;
}
