#include "ml.hpp"
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
using namespace fheco;

void cryptonets(
  const vector<size_t> &x_shape, const vector<size_t> &w1_shape, const vector<size_t> &b1_shape,
  const vector<size_t> &w4_shape, const vector<size_t> &b4_shape, const vector<size_t> &w8_shape,
  const vector<size_t> &b8_shape)
{
  // declare inputs
  int x_min_val = 0;
  int x_max_val = 1;
  int wb_min = -1;
  int wb_max = 1;
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

int main(int argc, char **argv)
{
  size_t vector_size = 4;
  if (argc > 1)
    vector_size = stoll(argv[1]);

  int trs_passes = 1;
  if (argc > 2)
    trs_passes = stoi(argv[2]);

  bool optimize = trs_passes > 0;

  cout << "vector_size: " << vector_size << ", "
       << "trs_passes: " << trs_passes << '\n';

  string func_name = "cryptonets";
  Compiler::create_func(func_name, vector_size, 20, true);

  cryptonets(
    vector<size_t>{28, 28, 1}, vector<size_t>{5, 5, 1, 5}, vector<size_t>{5}, vector<size_t>{5, 5, 5, 10},
    vector<size_t>{10}, vector<size_t>{40, 10}, vector<size_t>{10});

  // const auto &rand_inputs = Compiler::active_func()->inputs_info();
  // auto outputs = util::evaluate_on_clear(*Compiler::active_func(), rand_inputs);
  // if (outputs != Compiler::active_func()->outputs_info())
  //   throw logic_error("compilation correctness-test failed");

  // ofstream rand_example_os(func_name + "_rand_example.txt");
  // util::print_io_terms_values(*Compiler::active_func(), rand_example_os);

  // util::Quantifier quantifier(Compiler::active_func());
  // quantifier.run_analysis();
  // cout << Compiler::active_func()->inputs_info();
  // quantifier.print_info(cout);

  // util::ExprPrinter expr_printer(Compiler::active_func());
  // cout << "y " << expr_printer.expand_term(85158, util::ExprPrinter::Mode::infix, 100) << '\n';
  // cout << Compiler::active_func()->constants_values();

  // ofstream init_ir_os(func_name + "_init_ir.dot");
  // util::draw_ir(*Compiler::active_func(), init_ir_os);
  return 0;
}
