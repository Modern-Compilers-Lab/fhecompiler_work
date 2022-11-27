
#include "params_simulation.hpp"
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

using namespace std;
using namespace seal;

SEALContext bfv_params_simulation(
  int initial_plain_m_size, int xdepth, int safety_margin, bool use_least_levels, sec_level_type sec_level)
{
  if (initial_plain_m_size > 60)
    throw invalid_argument("plain modulus size too large");

  cout << "xdepth: " << xdepth << endl;

  string depth_too_large_msg = "xdepth too large, corresponding parameters not included in FHE security standard";
  scheme_type scheme = scheme_type::bfv;
  size_t max_poly_md = 32768;
  cout << "safety_margin: " << safety_margin << endl;
  cout << "Initial parameters using bfv_params_heuristic" << endl;
  // Set Initial parameters using using bfv_params_heuristic
  EncryptionParameters params = bfv_params_heuristic(
    initial_plain_m_size, xdepth, sec_level, use_least_levels, 1024, max_poly_md, depth_too_large_msg);
  // Initial coeff_m_bit_sizes
  vector<int> coeff_m_bit_sizes(params.coeff_modulus().size());
  int coeff_m_bit_count = 0;
  for (int i = 0; i < coeff_m_bit_sizes.size(); ++i)
  {
    coeff_m_bit_sizes[i] = params.coeff_modulus()[i].bit_count();
    coeff_m_bit_count += coeff_m_bit_sizes[i];
  }
  // Initial poly_modulus_degree
  size_t poly_modulus_degree = params.poly_modulus_degree();
  // Initial plain_modulus
  Modulus plain_modulus = params.plain_modulus();

  int test_count = 0;
  do
  {
    // Lambda function to increment parameters
    auto increment_params = [&]() {
      int max_coeff_m_bit_count = CoeffModulus::MaxBitCount(poly_modulus_degree, sec_level);
      if (coeff_m_bit_count < max_coeff_m_bit_count)
      {
        // Increment coeff modulus data level bit count
        int smallest_prime_idx = last_smallest_prime_index(coeff_m_bit_sizes);
        // Smallest prime size can be incremented
        if (coeff_m_bit_sizes[smallest_prime_idx] < 60)
        {
          // All primes have the same size
          if (smallest_prime_idx == coeff_m_bit_sizes.size() - 1)
          {
            // Not to increment only special prime size
            if (max_coeff_m_bit_count - coeff_m_bit_count > 1)
            {
              ++coeff_m_bit_sizes[smallest_prime_idx];
              ++coeff_m_bit_sizes[smallest_prime_idx - 1];
              coeff_m_bit_count += 2;
            }
            else
            {
              increase_poly_md(poly_modulus_degree, max_poly_md, depth_too_large_msg);
              plain_modulus =
                create_plain_modulus(poly_modulus_degree, plain_modulus.bit_count(), 60, depth_too_large_msg);
            }
          }
          else
          {
            ++coeff_m_bit_sizes[smallest_prime_idx];
            ++coeff_m_bit_count;
          }
        }
        // All prime sizes have reached 60, add a new prime
        else
        {
          int data_level_bc = coeff_m_bit_count - coeff_m_bit_sizes.back();
          // Remove special prime size
          coeff_m_bit_sizes.pop_back();
          // Increment the number of data level primes
          int data_level_nb_primes = coeff_m_bit_sizes.size() + 1;
          // coeff_m_bit_sizes with only data level primes sizes
          coeff_m_bit_sizes.assign(data_level_nb_primes, data_level_bc / data_level_nb_primes);
          // Add remaining bits and increment
          int remaining_bits = data_level_bc % coeff_m_bit_sizes.size();
          for (int i = 0; i < remaining_bits + 1; ++i)
            ++coeff_m_bit_sizes.end()[-1 - i];
          // Add special prime
          coeff_m_bit_sizes.push_back(coeff_m_bit_sizes.back());
          coeff_m_bit_count = data_level_bc + 1 + coeff_m_bit_sizes.back();
        }
      }
      else
      {
        increase_poly_md(poly_modulus_degree, max_poly_md, depth_too_large_msg);
        plain_modulus = create_plain_modulus(poly_modulus_degree, plain_modulus.bit_count(), 60, depth_too_large_msg);
      }
    };

    // Update EncryptionParameters object with current parameters
    vector<Modulus> coeff_modulus;
    try
    {
      coeff_modulus = CoeffModulus::Create(poly_modulus_degree, coeff_m_bit_sizes);
    }
    catch (logic_error &e)
    {
      // Not enough suitable primes could be found
      // cout << e.what() << endl;
      increment_params();
      continue;
    }
    params.set_poly_modulus_degree(poly_modulus_degree);
    params.set_coeff_modulus(coeff_modulus);
    params.set_plain_modulus(plain_modulus);
    // Create SEALContext
    SEALContext context(params, true, sec_level);
    // Test
    vector<int> noise_budgets;
    ++test_count;
    try
    {
      noise_budgets = test_params(context, xdepth);
    }
    catch (invalid_argument &e)
    {
      increment_params();
      continue;
    }
    // Print the first valid parameters
    if (test_count == 1)
      cout << "Valid initial parameters" << endl;
    else
      cout << "Valid parameters after testing " << test_count << " parameters sets" << endl;
    if (noise_budgets.back() <= safety_margin)
      cout << "Minimal parameters" << endl;
    else
      cout << "Not minimal parameters" << endl;
    print_parameters(context);
    // Print noise budget progress over the computation
    print_noise_budget_progress(noise_budgets);

    // Parameters are not minimal
    if (noise_budgets.back() > safety_margin)
    {
      // Reduce coeff modulus bit count
      reduce_coeff_modulus_bit_count(coeff_m_bit_sizes, coeff_m_bit_count, noise_budgets.back() - safety_margin);
      // Reduce poly_modulus_degree according to the minimal coeff_m_bit_count and sec_level
      while (CoeffModulus::MaxBitCount(poly_modulus_degree >> 1, sec_level) >= coeff_m_bit_count)
        poly_modulus_degree >>= 1;
      // If poly_modulus_degree is reduced
      if (params.poly_modulus_degree() > poly_modulus_degree)
      {
        params.set_poly_modulus_degree(poly_modulus_degree);
        // Retry to create plain modulus with size initial_plain_m_size
        plain_modulus = create_plain_modulus(
          poly_modulus_degree, initial_plain_m_size, params.plain_modulus().bit_count(),
          "Minimal parameters plain_modulus could not be created");
      }
      try
      {
        coeff_modulus = CoeffModulus::Create(poly_modulus_degree, coeff_m_bit_sizes);
      }
      catch (logic_error &e)
      {
        throw logic_error("Minimal parameters coeff_modulus primes could not be created");
      }
      params.set_coeff_modulus(coeff_modulus);
      params.set_plain_modulus(plain_modulus);
      context = SEALContext(params, true, sec_level);
      // Test parameters
      try
      {
        noise_budgets = test_params(context, xdepth);
      }
      catch (invalid_argument &e)
      {
        throw logic_error("Minimal parameters not valid, evaluation failed");
      }
      // Print minimal valid parameters
      cout << "---------------------------------------------------------------" << endl;
      cout << "Minimal parameters" << endl;
      print_parameters(context);
      // Print noise budget progress over the computation
      print_noise_budget_progress(noise_budgets);
    }
    return context;
  } while (true);
}

EncryptionParameters bfv_params_heuristic(
  int initial_plain_m_size, int xdepth, sec_level_type sec_level, bool use_least_levels, size_t initial_poly_md,
  size_t max_poly_md, const string &err_msg)
{
  if (initial_plain_m_size > 60)
    throw invalid_argument("plain modulus size too large");

  if (initial_poly_md > max_poly_md)
    throw invalid_argument("polynomial modulus too large, not included in fhe security standard");

  size_t poly_modulus_degree = initial_poly_md;
  vector<int> coeff_m_bit_sizes;
  Modulus plain_modulus = create_plain_modulus(poly_modulus_degree, initial_plain_m_size, 60, err_msg);

  do
  {
    // coeff_modulus data level (without special prime) bit count estimation
    int max_fresh_noise = 10;
    int poly_md_bit_count = util::get_significant_bit_count(poly_modulus_degree);
    int plain_m_size = plain_modulus.bit_count();
    int worst_case_mul_ng = poly_md_bit_count * 2 + plain_m_size + 1;
    int best_case_mul_ng = max(poly_md_bit_count, plain_m_size);
    // Calculate the average and round to the next integer
    int avg_mul_ng = ceil_float_div(worst_case_mul_ng + best_case_mul_ng, 2);
    int coeff_m_data_level_bc = plain_m_size + max_fresh_noise + xdepth * avg_mul_ng;
    // Set coeff_modulus primes sizes
    if (use_least_levels)
    {
      int lowest_nb_levels = ceil_float_div(coeff_m_data_level_bc, 60);
      coeff_m_bit_sizes.assign(lowest_nb_levels, 60);
      // Remove exceeding bits
      for (int i = 0; i < 60 - (coeff_m_data_level_bc % 60); ++i)
        --coeff_m_bit_sizes[i % coeff_m_bit_sizes.size()];
      // Add special prime
      coeff_m_bit_sizes.push_back(coeff_m_bit_sizes.back());
    }
    else
    {
      int nb_primes = xdepth + 1;
      int avg_prime_size = coeff_m_data_level_bc / nb_primes;
      coeff_m_bit_sizes.assign(nb_primes, avg_prime_size);
      // Add remaining bits
      for (int i = 0; i < coeff_m_data_level_bc % nb_primes; ++i)
        ++coeff_m_bit_sizes.end()[-1 - i];
      // Add special prime
      coeff_m_bit_sizes.push_back(coeff_m_bit_sizes.back());
    }

    int coeff_m_bit_count = coeff_m_data_level_bc + coeff_m_bit_sizes.back();
    // If for poly_modulus_degree and the given security level, coeff_modulus size cannot reach the needed value
    if (CoeffModulus::MaxBitCount(poly_modulus_degree, sec_level) < coeff_m_bit_count)
    {
      while (CoeffModulus::MaxBitCount(poly_modulus_degree, sec_level) < coeff_m_bit_count)
      {
        if (poly_modulus_degree == max_poly_md)
          throw invalid_argument(err_msg);

        // Increase poly_modulus_degree
        poly_modulus_degree <<= 1;
      }
      // Create plain modulus for batching with new poly_modulus_degree
      plain_modulus = create_plain_modulus(poly_modulus_degree, plain_m_size, 60, err_msg);
      // Repeat estimation with new poly_modulus_degree and plain_modulus
      continue;
    }
    if (xdepth)
      cout << "Estimated mul noise growth: " << avg_mul_ng << endl;
    break;

  } while (true);

  // Create coeff modulus primes using coeff_m_bit_sizes
  vector<Modulus> coeff_modulus = CoeffModulus::Create(poly_modulus_degree, coeff_m_bit_sizes);
  EncryptionParameters params(scheme_type::bfv);
  params.set_poly_modulus_degree(poly_modulus_degree);
  params.set_coeff_modulus(coeff_modulus);
  params.set_plain_modulus(plain_modulus);
  return params;
}

vector<int> test_params(const SEALContext &context, int xdepth)
{
  BatchEncoder batch_encoder(context);
  KeyGenerator keygen(context);
  SecretKey secret_key = keygen.secret_key();
  PublicKey public_key;
  keygen.create_public_key(public_key);
  RelinKeys relin_keys;
  keygen.create_relin_keys(relin_keys);
  Encryptor encryptor(context, public_key);
  Evaluator evaluator(context);
  Decryptor decryptor(context, secret_key);

  size_t slot_count = batch_encoder.slot_count();
  vector<uint64_t> pod_vector;
  random_device rd;
  for (size_t i = 0; i < slot_count; ++i)
    pod_vector.push_back(context.first_context_data()->parms().plain_modulus().reduce(rd()));

  Plaintext plain(context.first_context_data()->parms().poly_modulus_degree(), 0);
  batch_encoder.encode(pod_vector, plain);
  Ciphertext encrypted(context);
  encryptor.encrypt(plain, encrypted);

  vector<int> noise_budgets(xdepth + 1);
  noise_budgets[0] = decryptor.invariant_noise_budget(encrypted);
  if (noise_budgets[0] == 0)
    throw invalid_argument("insufficient noise budget");
  for (int i = 0; i < xdepth; ++i)
  {
    evaluator.multiply_inplace(encrypted, encrypted);
    evaluator.relinearize_inplace(encrypted, relin_keys);
    int noise_budget = decryptor.invariant_noise_budget(encrypted);
    if (noise_budget == 0)
      throw invalid_argument("insufficient noise budget");
    noise_budgets[i + 1] = noise_budget;
  }
  return noise_budgets;
}

void increase_poly_md(size_t &poly_modulus_degree, size_t max_poly_md, const string &err_msg)
{
  if (poly_modulus_degree == max_poly_md)
    throw invalid_argument(err_msg);

  poly_modulus_degree <<= 1;
}

Modulus create_plain_modulus(size_t poly_modulus_degree, int initial_size, int max_size, const string &err_msg)
{
  if (initial_size > max_size)
    throw invalid_argument("plain_modulus initial size greater than max_size");

  do
  {
    try
    {
      return PlainModulus::Batching(poly_modulus_degree, initial_size);
    }
    // Suitable prime could not be found
    catch (logic_error &e)
    {
      if (initial_size == max_size)
        throw invalid_argument(err_msg);
      ++initial_size;
    }
  } while (true);
}

void reduce_coeff_modulus_bit_count(vector<int> &bit_sizes, int &bit_count, int amount)
{
  // Calculate data level bit count
  int data_level_bit_count = bit_count - bit_sizes.back();
  // Remove special prime size
  bit_sizes.pop_back();
  // Find the first bigger prime index
  int idx = first_biggest_prime_index(bit_sizes);
  // Reduce data level primes bit sizes by amount starting from idx
  for (int i = 0; i < amount; ++i)
    --bit_sizes[(idx + i) % bit_sizes.size()];
  data_level_bit_count -= amount;
  // Add special prime
  bit_sizes.push_back(bit_sizes.back());
  // Update bit_count
  bit_count = data_level_bit_count + bit_sizes.back();
}

int first_biggest_prime_index(const vector<int> &bit_sizes)
{
  int idx = bit_sizes.size() - 1;
  while (idx > 0 && bit_sizes[idx] == bit_sizes[idx - 1])
    --idx;
  return idx;
}

int last_smallest_prime_index(const vector<int> &bit_sizes)
{
  int idx = 0;
  while (idx < bit_sizes.size() - 1 && bit_sizes[idx] == bit_sizes[idx + 1])
    ++idx;
  return idx;
}

void print_noise_budget_progress(const vector<int> &noise_budgets)
{
  cout << "Fresh noise budget: " << noise_budgets[0] << endl;
  int nb_muls = noise_budgets.size() - 1;
  for (int i = 1; i < nb_muls + 1; ++i)
    cout << "After " << i << " seq muls: " << noise_budgets[i] << endl;
  if (nb_muls)
  {
    int total_noise_growth = noise_budgets[0] - noise_budgets.back();
    cout << "Observed mul avg noise growth: " << ceil_float_div(total_noise_growth, nb_muls) << endl;
  }
}

void print_parameters(const seal::SEALContext &context)
{
  auto &context_data = *context.key_context_data();

  /*
  Which scheme are we using?
  */
  std::string scheme_name;
  switch (context_data.parms().scheme())
  {
  case seal::scheme_type::bfv:
    scheme_name = "BFV";
    break;
  case seal::scheme_type::ckks:
    scheme_name = "CKKS";
    break;
  case seal::scheme_type::bgv:
    scheme_name = "BGV";
    break;
  default:
    throw std::invalid_argument("unsupported scheme");
  }
  std::cout << "/" << std::endl;
  std::cout << "| Encryption parameters :" << std::endl;
  std::cout << "|   scheme: " << scheme_name << std::endl;
  std::cout << "|   poly_modulus_degree: " << context_data.parms().poly_modulus_degree() << std::endl;

  /*
  Print the size of the true (product) coefficient modulus.
  */
  std::cout << "|   coeff_modulus size: ";
  std::cout << context_data.total_coeff_modulus_bit_count() << " (";
  auto coeff_modulus = context_data.parms().coeff_modulus();
  std::size_t coeff_modulus_size = coeff_modulus.size();
  for (std::size_t i = 0; i < coeff_modulus_size - 1; ++i)
  {
    std::cout << coeff_modulus[i].bit_count() << " + ";
  }
  std::cout << coeff_modulus.back().bit_count();
  std::cout << ") bits" << std::endl;

  /*
  For BFV and BGV schemes print the plain_modulus parameter.
  */
  if (
    context_data.parms().scheme() == seal::scheme_type::bfv || context_data.parms().scheme() == seal::scheme_type::bgv)
  {
    std::cout << "|   plain_modulus: " << context_data.parms().plain_modulus().value() << std::endl;
  }

  std::cout << "\\" << std::endl;
}

int ceil_float_div(int a, int b)
{
  return static_cast<int>(ceilf(static_cast<float>(a) / b));
}
