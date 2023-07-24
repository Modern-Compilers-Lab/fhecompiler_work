#include "fheco/fheco.hpp"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <ostream>
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
  vector<vector<vector<Ciphertext>>> x;
  for (size_t i = 0; i < x_shape[0]; ++i)
  {
    x.push_back(vector<vector<Ciphertext>>());
    for (size_t j = 0; j < x_shape[1]; ++j)
    {
      x[i].push_back(vector<Ciphertext>());
      for (size_t k = 0; k < x_shape[2]; ++k)
        x[i][j].push_back(Ciphertext("x[" + to_string(i) + "][" + to_string(j) + "][" + to_string(k) + "]"));
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
          w1[i][j][k].push_back(
            Plaintext("w1[" + to_string(i) + "][" + to_string(j) + "][" + to_string(k) + "][" + to_string(l) + "]"));
      }
    }
  }

  vector<Plaintext> b1;
  for (size_t i = 0; i < b1_shape[0]; ++i)
    b1.push_back(Plaintext("b1[" + to_string(i) + "]"));

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
          w4[i][j][k].push_back(
            Plaintext("w4[" + to_string(i) + "][" + to_string(j) + "][" + to_string(k) + "][" + to_string(l) + "]"));
      }
    }
  }

  vector<Plaintext> b4;
  for (size_t i = 0; i < b4_shape[0]; ++i)
    b4.push_back(Plaintext("b4[" + to_string(i) + "]"));

  vector<vector<Plaintext>> w8;
  for (size_t i = 0; i < w8_shape[0]; ++i)
  {
    w8.push_back(vector<Plaintext>());
    for (size_t j = 0; j < w8_shape[1]; ++j)
      w8[i].push_back(Plaintext("w8[" + to_string(i) + "][" + to_string(j) + "]"));
  }

  vector<Plaintext> b8;
  for (size_t i = 0; i < b8_shape[0]; ++i)
    b8.push_back(Plaintext("b8[" + to_string(i) + "]"));

  // predict
  auto y = predict(x, w1, b1, w4, b4, w8, b8);
  // declare outputs
  for (size_t i = 0; i < y.size(); ++i)
    y[i].set_output("y[" + to_string(i) + "]");
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

  bool const_folding = true;
  if (argc > 5)
    const_folding = stoi(argv[5]);

  print_bool_arg(call_quantifier, "quantifier", clog);
  clog << " ";
  clog << ruleset << "_trs";
  clog << " ";
  clog << rewrite_heuristic;
  clog << " ";
  print_bool_arg(cse, "cse", clog);
  clog << " ";
  print_bool_arg(const_folding, "constant_folding", clog);
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

  chrono::high_resolution_clock::time_point t;
  chrono::duration<double, milli> elapsed;
  t = chrono::high_resolution_clock::now();
  string func_name = "cryptonets";
  const auto &func = Compiler::create_func(func_name, 8192, 17, true, false);

  cryptonets(
    vector<size_t>{28, 28, 1}, vector<size_t>{5, 5, 1, 5}, vector<size_t>{5}, vector<size_t>{5, 5, 5, 10},
    vector<size_t>{10}, vector<size_t>{40, 10}, vector<size_t>{10});

  string gen_name = "_gen_he_" + func_name;
  string gen_path = "he/" + gen_name;
  ofstream header_os(gen_path + ".hpp");
  ofstream source_os(gen_path + ".cpp");

  Compiler::compile(func, ruleset, rewrite_heuristic, header_os, gen_name + ".hpp", source_os);

  elapsed = chrono::high_resolution_clock::now() - t;
  cout << elapsed.count() << " ms\n";

  if (call_quantifier)
  {
    util::Quantifier quantifier{func};
    quantifier.run_all_analysis();
    quantifier.print_info(cout);
  }

  return 0;
}
