#include "fhecompiler/fhecompiler.hpp"
#include <bits/stdc++.h>

inline fhecompiler::Ciphertext sum_all_slots(fhecompiler::Ciphertext &x, int vector_size)
{
  fhecompiler::Ciphertext result = x;
  int step = vector_size - 1;
  for (; step > 0;)
  {
    result += (x << (step--));
  }
  // result of sum will be in the first slot
  return result;
}

inline fhecompiler::Ciphertext sum_all_slots_log2(const fhecompiler::Ciphertext &x, size_t vector_size)
{
  fhecompiler::Ciphertext result = x;

  auto clog2 = [](int32_t x) -> int32_t {
    int32_t r = 0;
    while (x > 1)
    {
      x /= 2;
      r += 1;
    }
    return r;
  };

  /*
    A : 8192 x 64
    B : 64 x 256
    n = poly_modulus_degree = 8192/64 = 128

    1 1st 128 lines
    1 2nd 128 lines

    n = 8192

    L is a line of A', L = [1,2,4,5|6,4,3,2|.....] sum of segments

    A'(A packed) : 64 x 64
    B = 64 x 256
    C = A' x B
  */

  auto next_power_of_2 = [&clog2](size_t n) -> size_t {
    if (__builtin_popcount(n) == 1)
      return n;

    auto log2_of_n = clog2(n);

    // I assume no overflow as n will be in range [2^10, 2^16]
    return (1 << (log2_of_n + 1));
  };

  vector_size = next_power_of_2(vector_size);
  int32_t max_num_steps = clog2(vector_size);

  int32_t rot_step = 1;
  for (; max_num_steps--; rot_step *= 2)
  {
    fhecompiler::Ciphertext new_rotated_cipher = result << rot_step;
    result += new_rotated_cipher;
  }
  // result of sum will be in the first slot
  return result;
}

int main()
{
  try
  {
    fhecompiler::init("reduction", fhecompiler::Scheme::bfv, fhecompiler::Backend::SEAL);

    fhecompiler::Ciphertext ct1("ct1", fhecompiler::VarType::input);

    fhecompiler::Ciphertext slots_sum("slots_sum", fhecompiler::VarType::temp);
    fhecompiler::Ciphertext dummy_output("ct_otuput", fhecompiler::VarType::output);

    slots_sum = sum_all_slots(ct1, 1025);

    dummy_output = slots_sum;

    size_t polynomial_modulus_degree = 4096;
    size_t plaintext_modulus = 786433;

    params_selector::EncryptionParameters params;
    params.set_plaintext_modulus(plaintext_modulus);
    params.set_polynomial_modulus_degree(polynomial_modulus_degree);

    fhecompiler::compile("reduction.hpp", &params);
  }
  catch (const char *emessage)
  {
    std::cout << emessage << '\n';
  }
  return 0;
}
