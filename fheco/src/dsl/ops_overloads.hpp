#pragma once

#include "ciphertext.hpp"
#include "datatypes_const.hpp"
#include "datatypes_util.hpp"
#include "plaintext.hpp"
#include "scalar.hpp"

namespace fhecompiler
{

// addition
Ciphertext &operator+=(Ciphertext &lhs, const Plaintext &rhs);
Ciphertext operator+(const Ciphertext &lhs, const Plaintext &rhs);
Ciphertext operator+(const Plaintext &lhs, const Ciphertext &rhs);

/*
Plaintext operator+(const Plaintext &lhs, const Scalar &rhs);
Plaintext &operator+=(Plaintext &lhs, const Scalar &rhs);
Plaintext operator+(const Scalar &rhs, const Plaintext &lhs);
*/

Ciphertext operator+(const Ciphertext &lhs, const Scalar &rhs);
Ciphertext &operator+=(Ciphertext &lhs, const Scalar &rhs);
Ciphertext operator+(const Scalar &lhs, const Ciphertext &rhs);
// subtraction
Ciphertext operator-(const Ciphertext &lhs, const Plaintext &rhs);
Ciphertext &operator-=(Ciphertext &lhs, const Plaintext &rhs);

/*
Ciphertext operator-(const Plaintext &lhs, const Ciphertext &rhs);
*/

/*
Plaintext operator-(const Plaintext &lhs, const Scalar &rhs);
Plaintext &operator-=(Plaintext &lhs, const Scalar &rhs);

Plaintext operator-(const Scalar &lhs, const Plaintext &rhs);
*/
Ciphertext operator-(const Ciphertext &lhs, const Scalar &rhs);
Ciphertext operator-=(Ciphertext &lhs, const Scalar &rhs);

/*
Ciphertext operator-(const Scalar &lhs, const Ciphertext &rhs);
*/

// multiplication
Ciphertext operator*(const Ciphertext &lhs, const Plaintext &rhs);

Ciphertext &operator*=(Ciphertext &lhs, const Plaintext &rhs);
Ciphertext operator*(const Plaintext &lhs, const Ciphertext &rhs);

Ciphertext operator*(const Ciphertext &lhs, const Scalar &rhs);

Ciphertext operator*(const Scalar &lhs, const Ciphertext &rhs);
Ciphertext &operator*=(Ciphertext &lhs, const Scalar &rhs);
/*

Plaintext operator*(const Scalar &lhs, const Plaintext &rhs);
Plaintext &operator*=(Plaintext &lhs, const Scalar &rhs);
Plaintext operator*(const Plaintext &lhs, const Scalar &rhs);
*/

} // namespace fhecompiler