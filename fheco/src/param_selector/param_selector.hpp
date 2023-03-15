#pragma once

#include "program.hpp"
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace param_selector
{

class ParameterSelector
{
public:
  ParameterSelector(ir::Program *program) : program_(program) {}

  void select_params(bool use_mod_switch = true);

private:
  struct NoiseEstimatesValue
  {
    int fresh_noise;
    int mul_noise_growth;
    int mul_plain_noise_growth;
  };

  ir::Program *program_;

  static std::unordered_map<fhecompiler::SecurityLevel, std::unordered_map<std::size_t, int>> security_standard;

  static std::map<int, std::map<std::size_t, NoiseEstimatesValue>> bfv_noise_estimates_seal;

  void select_params_bfv(bool use_mod_switch = true);

  int simulate_noise_bfv(
    NoiseEstimatesValue noise_estimates_value, std::unordered_map<std::string, int> &nodes_noise) const;

  void insert_mod_switch_bfv(
    const std::vector<int> &data_level_primes_sizes, std::unordered_map<std::string, int> &nodes_noise,
    int safety_margin);

  std::unordered_map<std::string, std::unordered_set<std::string>> get_outputs_composing_nodes() const;
};
} // namespace param_selector
