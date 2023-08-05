#include <cstddef>
#include <cstdint>
#include <utility>
#include "gen_he_sobel_noopt.hpp"

using namespace std;
using namespace seal;

void sobel_noopt(const unordered_map<string, Ciphertext> &encrypted_inputs,
const unordered_map<string, Plaintext> &encoded_inputs,
unordered_map<string, Ciphertext> &encrypted_outputs,
unordered_map<string, Plaintext> &encoded_outputs,
const BatchEncoder &encoder,
const Encryptor &encryptor,
const Evaluator &evaluator,
const RelinKeys &relin_keys,
const GaloisKeys &galois_keys)
{
Ciphertext c1 = encrypted_inputs.at("img");
size_t slot_count = encoder.slot_count();
Plaintext p52;
encoder.encode(vector<std::int64_t>(slot_count, 0), p52);
Plaintext p62;
encoder.encode(vector<std::int64_t>(slot_count, 1), p62);
Plaintext p4;
encoder.encode(vector<std::int64_t>(slot_count, -1), p4);
Plaintext p59;
encoder.encode(vector<std::int64_t>(slot_count, 2), p59);
Plaintext p30;
encoder.encode(vector<std::int64_t>(slot_count, 1), p30);
Plaintext p49;
encoder.encode(vector<std::int64_t>(slot_count, 0), p49);
Plaintext p20;
encoder.encode(vector<std::int64_t>(slot_count, 2), p20);
Plaintext p17;
encoder.encode(vector<std::int64_t>(slot_count, 0), p17);
Plaintext p46;
encoder.encode(vector<std::int64_t>(slot_count, 0), p46);
Plaintext p7;
encoder.encode(vector<std::int64_t>(slot_count, 0), p7);
Plaintext p36;
encoder.encode(vector<std::int64_t>(slot_count, -1), p36);
Plaintext p14;
encoder.encode(vector<std::int64_t>(slot_count, -2), p14);
Plaintext p56;
encoder.encode(vector<std::int64_t>(slot_count, 1), p56);
Plaintext p27;
encoder.encode(vector<std::int64_t>(slot_count, 0), p27);
Plaintext p24;
encoder.encode(vector<std::int64_t>(slot_count, -1), p24);
Plaintext p10;
encoder.encode(vector<std::int64_t>(slot_count, 1), p10);
Plaintext p39;
encoder.encode(vector<std::int64_t>(slot_count, -2), p39);
Plaintext p42;
encoder.encode(vector<std::int64_t>(slot_count, -1), p42);
Ciphertext c2;
evaluator.rotate_rows(c1, 4032, galois_keys, c2);
Ciphertext c5;
evaluator.rotate_rows(c2, 4095, galois_keys, c5);
evaluator.multiply_plain(c5, p4, c5);
Ciphertext c74;
evaluator.multiply_plain(c2, p7, c74);
evaluator.add(c5, c74, c5);
evaluator.rotate_rows(c2, 1, galois_keys, c74);
evaluator.multiply_plain(c74, p10, c74);
evaluator.add(c5, c74, c5);
evaluator.rotate_rows(c1, 4095, galois_keys, c74);
evaluator.multiply_plain(c74, p14, c74);
Ciphertext c77;
evaluator.multiply_plain(c1, p17, c77);
evaluator.add(c74, c77, c74);
evaluator.rotate_rows(c1, 1, galois_keys, c77);
evaluator.multiply_plain(c77, p20, c77);
evaluator.add(c74, c77, c74);
evaluator.add(c5, c74, c5);
evaluator.rotate_rows(c1, 64, galois_keys, c74);
evaluator.rotate_rows(c74, 4095, galois_keys, c77);
evaluator.multiply_plain(c77, p24, c77);
Ciphertext c80;
evaluator.multiply_plain(c74, p27, c80);
evaluator.add(c77, c80, c77);
evaluator.rotate_rows(c74, 1, galois_keys, c80);
evaluator.multiply_plain(c80, p30, c80);
evaluator.add(c77, c80, c77);
evaluator.add(c5, c77, c5);
evaluator.multiply(c5, c5, c5);
evaluator.relinearize(c5, relin_keys, c5);
evaluator.rotate_rows(c2, 4095, galois_keys, c77);
evaluator.multiply_plain(c77, p36, c77);
evaluator.multiply_plain(c2, p39, c80);
evaluator.add(c77, c80, c77);
evaluator.rotate_rows(c2, 1, galois_keys, c2);
evaluator.multiply_plain(c2, p42, c2);
evaluator.add(c77, c2, c77);
evaluator.rotate_rows(c1, 4095, galois_keys, c80);
evaluator.multiply_plain(c80, p46, c80);
evaluator.multiply_plain(c1, p49, c2);
evaluator.add(c80, c2, c80);
evaluator.rotate_rows(c1, 1, galois_keys, c1);
evaluator.multiply_plain(c1, p52, c1);
evaluator.add(c80, c1, c80);
evaluator.add(c77, c80, c77);
evaluator.rotate_rows(c74, 4095, galois_keys, c1);
evaluator.multiply_plain(c1, p56, c1);
evaluator.multiply_plain(c74, p59, c2);
evaluator.add(c1, c2, c1);
evaluator.rotate_rows(c74, 1, galois_keys, c74);
evaluator.multiply_plain(c74, p62, c74);
evaluator.add(c1, c74, c1);
evaluator.add(c77, c1, c77);
evaluator.multiply(c77, c77, c77);
evaluator.relinearize(c77, relin_keys, c77);
evaluator.add(c5, c77, c5);
encrypted_outputs.emplace("result", move(c5));
}

vector<int> get_rotation_steps_sobel_noopt(){
return vector<int>{4032, 4095, 1, 64};
}