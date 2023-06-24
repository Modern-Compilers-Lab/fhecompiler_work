#include "gen_he_box_blur.hpp"
#include <cstddef>
#include <cstdint>
#include <utility>

using namespace std;
using namespace seal;

void box_blur(
  const unordered_map<string, Ciphertext> &encrypted_inputs, const unordered_map<string, Plaintext> &encoded_inputs,
  unordered_map<string, Ciphertext> &encrypted_outputs, unordered_map<string, Plaintext> &encoded_outputs,
  const BatchEncoder &encoder, const Encryptor &encryptor, const Evaluator &evaluator, const RelinKeys &relin_keys,
  const GaloisKeys &galois_keys)
{
  Ciphertext c1 = encrypted_inputs.at("c0");
  size_t slot_count = encoder.slot_count();
  Ciphertext c2;
  evaluator.rotate_rows(c1, 1, galois_keys, c2);
  evaluator.add(c1, c2, c1);
  evaluator.rotate_rows(c1, 5, galois_keys, c2);
  evaluator.add(c1, c2, c1);
  encrypted_outputs.emplace("c6", move(c1));
}

vector<int> get_rotation_steps_box_blur()
{
  return vector<int>{1, 5};
}
