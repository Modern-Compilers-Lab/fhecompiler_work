#include "fhecompiler.hpp"
#include <bits/stdc++.h>

using namespace std;

using namespace fhecompiler;

Ciphertext sum(const Ciphertext &a, const Ciphertext &b)
{
  return a + b;
}

void print_string(std::optional<std::string> string_opt)
{
  if (string_opt != nullopt)
    std::cout << *string_opt << "\n";
}

int main()
{

  /*
    Encrypt, Decrypt, Evaluate(add, mul, sub, rotate), generate keys
  */

  try
  {
    fhecompiler::init("test1", 2 << 14, fhecompiler::Scheme::bfv);

    fhecompiler::Ciphertext ct1("ct1", VarType::input);

    fhecompiler::Plaintext pt1("pt1", VarType::input);

    fhecompiler::Ciphertext output1("output1", VarType::output);

    // fhecompiler::params.set_coef_modulus({50, 50, 50});
    fhecompiler::params.set_plaintext_modulus(786433);
    // fhecompiler::params.set_plaintext_modulus_bit_length(20);
    fhecompiler::params.set_polynomial_modulus_degree(2 << 14);

    output1 = ct1 + pt1;

    output1 >>= 12;

    output1 *= 13244;

    fhecompiler::compile("test1.hpp");
  }
  catch (const char *message)
  {
    std::cout << message << "\n";
  }

  // initial noise_budget
  return 0;
}
