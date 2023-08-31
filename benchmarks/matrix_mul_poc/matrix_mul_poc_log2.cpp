#include "fhecompiler/fhecompiler.hpp"

inline fhecompiler::Ciphertext sum_all_slots(fhecompiler::Ciphertext &x, int vector_size)
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

/*
inline fhecompiler::Ciphertext sum_all_slots(fhecompiler::Ciphertext &x, int number_slots_to_rotate)
{
  fhecompiler::Ciphertext result = x;
  int step = number_slots_to_rotate - 1;
  for (; step-- > 0;)
  {
    // fhecompiler::Ciphertext cipher_rotated = rotated_ciphers.back() << 1;
    x <<= 1;
    result += x;
  }
  // result of sum will be in the first slot
  x <<= 1;
  return result;
}
*/

inline fhecompiler::Ciphertext sum_all_slots2(const fhecompiler::Ciphertext &x, size_t vector_size)
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
    fhecompiler::init("matrix_mul_poc_log2", fhecompiler::Scheme::bfv, fhecompiler::Backend::SEAL);

    size_t polynomial_modulus_degree = 4096;
    size_t plaintext_modulus = 786433;

    std::vector<std::vector<int64_t>> A; // = {{1, 2, 3, -2}, {-5, 3, 2, 0}, {1, 0, 1, -3}, {5, 3, 2, 0}, {5, 3, 2, 0}};
    std::vector<std::vector<int64_t>> B; // = {{0, 1, 9}, {-7, -10, 2}, {1, 9, 0}, {-8, 2, 18}};

    const int N = 64;
    const int M = 128;
    const int P = 128;
    const int Q = 8;

    for (size_t i = 0; i < N; i++)
    {
      std::vector<int64_t> line;
      for (size_t j = 0; j < M; j++)
      {
        line.push_back((i + 1) * (j + 1));
      }
      A.push_back(line);
    }
    for (size_t i = 0; i < P; i++)
    {
      std::vector<int64_t> line;
      for (size_t j = 0; j < Q; j++)
      {
        line.push_back((i + 1) * (j + 1));
      }
      B.push_back(line);
    }

    std::vector<fhecompiler::Ciphertext> A_encrypted;
    // encoding and encryption of A
    size_t vector_size =
      8192; // this the number of slots from user perspective, compiler later may select n > vector_size
    size_t nb_lines_to_pack_in_one_ciphertext = std::min(A.size(), vector_size / A[0].size());
    std::cout << nb_lines_to_pack_in_one_ciphertext << "\n";
    /*
      now nb_lines_to_pack_in_one_ciphertext is the number of lines that needs to be flattened and packed in one
      plaintext to encrypt
    */
    for (size_t i = 0; i < A.size(); i += nb_lines_to_pack_in_one_ciphertext)
    {
      std::vector<int64_t> lines_flattened;
      for (size_t k = 0; k < nb_lines_to_pack_in_one_ciphertext; k++)
      {
        for (size_t j = 0; j < A[0].size(); j++)
        {
          lines_flattened.push_back(A[i + k][j]);
        }
      }
      std::vector<int64_t> lines_flattened2;
      size_t nb_iter = 8192 / vector_size;
      // fhecompiler::Ciphertext input("input" + std::to_string(i), fhecompiler::VarType::input);
      for (size_t iter = 0; iter < nb_iter; iter++)
      {
        for (auto &x : lines_flattened)
          lines_flattened2.push_back(x);
      }
      fhecompiler::Plaintext lines_flattened_pt(lines_flattened);
      // A_encrypted.push_back(fhecompiler::Ciphertext::encrypt(lines_flattened2));
      A_encrypted.push_back(fhecompiler::Ciphertext::encrypt(lines_flattened));
      // A_encrypted.push_back(input);
    }
    std::cout << "A encrypted ...\n";
    //  encrypt by column for matrix B
    std::vector<fhecompiler::Plaintext> B_transpose;
    for (size_t column_index = 0; column_index < B[0].size(); column_index++)
    {
      std::vector<int64_t> column_data;
      std::vector<int64_t> column_duplicated;
      for (size_t i = 0; i < B.size(); i++)
      {
        column_data.push_back(B[i][column_index]);
      }
      for (size_t k = 0; k < nb_lines_to_pack_in_one_ciphertext; k++)
      {
        for (auto &v : column_data)
          column_duplicated.push_back(v);
      }
      // fhecompiler::Ciphertext column_encrypted = fhecompiler::Ciphertext::encrypt(column_duplicated);
      std::vector<int64_t> column_duplicated2;
      size_t nb_iter = 8192 / (vector_size);
      for (size_t iter = 0; iter < nb_iter; iter++)
      {
        for (auto &x : column_duplicated)
          column_duplicated2.push_back(x);
      }
      // fhecompiler::Plaintext clmn_pt("pt_input_" + std::to_string(column_index), fhecompiler::VarType::input);
      // fhecompiler::Plaintext clmn_pt(column_duplicated2);
      fhecompiler::Plaintext clmn_pt(column_duplicated);
      B_transpose.push_back(clmn_pt);
    }
    std::cout << "B encoded ...\n";
    // C contains result of multiplication
    std::vector<fhecompiler::Ciphertext> C_encrypted;
    // make outputs
    std::vector<fhecompiler::Ciphertext> outputs;
    std::vector<std::vector<fhecompiler::Ciphertext>> lines_of_C_encrypted(A.size());
    std::vector<int64_t> mask(8192, 0);
    mask[0] = 1;
    fhecompiler::Plaintext mask_pt(mask);

    for (int i = 0; i < A_encrypted.size(); i++) // 8
    {
      for (int j = 0; j < B_transpose.size(); j++) // 256
      {
        fhecompiler::Ciphertext simd_product = A_encrypted[i] * B_transpose[j];
        /*
          making outputs
          simd_product is now something like [X,X,X,X,X,Y,Y,Y,Y,Y,Z,Z,Z,Z,Z, .....]
          we want to sum up X's with each other and the same thing for Z's and Y's. Same letter means that elements will
          be used to create same entry in result matrix, frequency of each letter is equal to B.size() or A[0].size()
        */
        for (size_t k = 0; k < nb_lines_to_pack_in_one_ciphertext; k++) // 8192
        {
          size_t number_of_slots_to_sum = A[0].size(); // = B.size()
          std::vector<int64_t> shift_mask(vector_size, 0);
          for (size_t p = 0; p < A[0].size(); p++)
          {
            shift_mask[k * A[0].size() + p] = 1;
          }
          size_t nb_iter = 8192 / vector_size;
          std::vector<int64_t> shift_mask_dup;
          for (size_t l = 0; l < nb_iter; l++)
          {
            for (auto &x : shift_mask)
              shift_mask_dup.push_back(x);
          }
          // fhecompiler::Plaintext shift_mask_pt(shift_mask_dup);
          fhecompiler::Plaintext shift_mask_pt(shift_mask);
          fhecompiler::Ciphertext simd_product_masked = simd_product * shift_mask_pt;
          fhecompiler::Ciphertext slots_sum = sum_all_slots(simd_product_masked, vector_size);
          fhecompiler::Ciphertext cipher_with_first_slot_only = slots_sum * mask_pt;
          size_t corresponding_line = i * nb_lines_to_pack_in_one_ciphertext + k;
          lines_of_C_encrypted[corresponding_line].push_back(cipher_with_first_slot_only);
        }
      }
    }
    std::cout << "dot products done ...\n";
    // making outputs
    for (int i = 0; i < lines_of_C_encrypted.size(); i++)
    {
      fhecompiler::Ciphertext output_line("output" + std::to_string(i), fhecompiler::VarType::output);
      output_line = lines_of_C_encrypted[i][0];
      for (int j = 1; j < lines_of_C_encrypted[i].size(); j++)
      {
        fhecompiler::Ciphertext lines_of_C_encrypted_rotated;
        output_line += (lines_of_C_encrypted[i][j] >> j);
      }
      C_encrypted.push_back(output_line);
    }
    std::cout << "lines computation done ...\n";

    std::cout << "number of lines in C : " << lines_of_C_encrypted.size() << "\n";

    params_selector::EncryptionParameters params;
    params.set_plaintext_modulus(plaintext_modulus);
    params.set_polynomial_modulus_degree(polynomial_modulus_degree);

    fhecompiler::compile("matrix_mul_poc_log2.hpp", &params);
  }
  catch (const char *message)
  {
    std::cout << message << "\n";
  }

  return 0;
}