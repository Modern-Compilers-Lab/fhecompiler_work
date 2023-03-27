#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#define MOD_BIT_COUNT_MAX 60

namespace param_selector
{
class EncryptionParameters
{
public:
  EncryptionParameters() {}

  EncryptionParameters(std::size_t poly_modulus_degree, int plain_modulus_bit_size, int coeff_mod_data_level_bit_count);

  int increase_coeff_mod_bit_sizes(int max_total_amount);

  int last_coeff_mod_big_prime_idx() const;

  inline int poly_modulus_degree() const { return poly_modulus_degree_; }

  inline int plain_modulus_bit_size() const { return plain_modulus_bit_size_; }

  inline const std::vector<int> &coeff_mod_bit_sizes() const { return coeff_mod_bit_sizes_; }

  std::vector<int> coeff_mod_data_level_bit_sizes() const;

  inline int coeff_mod_bit_count() const { return coeff_mod_bit_count_; }

  inline int coeff_mod_data_level_bit_count() const { return coeff_mod_bit_count_ - coeff_mod_bit_sizes_.back(); }

  void print_parameters() const;

private:
  std::size_t poly_modulus_degree_ = 0;
  int plain_modulus_bit_size_ = 0;
  std::vector<int> coeff_mod_bit_sizes_;
  int coeff_mod_bit_count_ = 0;
};
} // namespace param_selector
