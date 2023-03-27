#pragma once

#include "seal/seal.h"
#include <cstddef>
#include <map>
#include <string>

#define MOD_BIT_COUNT_MAX 60

struct OpsNoiseGrowth
{
  int encrypt = 0;
  int mul = 0;
  int mul_plain = 0;
  int add = 0;
  int add_plain = 0;
  int rotate = 0;
  int relin = 0;
};

using bfv_noise_experiments_map = std::map<int, std::map<std::size_t, OpsNoiseGrowth>>;

seal::EncryptionParameters bfv_no_security_params_heuristic(std::size_t poly_mod, int init_plain_mod_size, int xdepth);

seal::Modulus create_plain_mod(std::size_t poly_mod, int plain_mod_init_size);

OpsNoiseGrowth estimate_noise_growth_bfv(const seal::SEALContext &context, int xdepth);

void serialize_bfv_noise_experiments(
  const bfv_noise_experiments_map &bfv_noise_experiments, int repeat, int xdepth, const std::string &file_name);

void print_parameters(const seal::SEALContext &context);
