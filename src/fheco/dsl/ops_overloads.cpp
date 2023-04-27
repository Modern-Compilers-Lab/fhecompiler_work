#include "fheco/dsl/ops_overloads.hpp"
#include "fheco/dsl/compiler.hpp"
#include "fheco/ir/op_code.hpp"
#include <cstddef>
#include <tuple>
#include <vector>

using namespace std;

namespace fheco
{
// addition
Ciphertext operator+(const Ciphertext &lhs, const Ciphertext &rhs)
{
  if (lhs.idx().size() || rhs.idx().size())
    throw invalid_argument("subscript read must be performed on const variables");

  Ciphertext dest{};
  Compiler::active_func()->operate_binary(ir::OpCode::add, lhs, rhs, dest);
  return dest;
}

Ciphertext operator+(const Ciphertext &lhs, const Plaintext &rhs)
{
  if (lhs.idx().size() || rhs.idx().size())
    throw invalid_argument("subscript read must be performed on const variables");

  Ciphertext dest{};
  Compiler::active_func()->operate_binary(ir::OpCode::add, lhs, rhs, dest);
  return dest;
}

Ciphertext operator+(const Ciphertext &lhs, const Scalar &rhs)
{
  if (lhs.idx().size())
    throw invalid_argument("subscript read must be performed on const variables");

  Ciphertext dest{};
  Compiler::active_func()->operate_binary(ir::OpCode::add, lhs, rhs, dest);
  return dest;
}

Ciphertext operator+(const Plaintext &lhs, const Ciphertext &rhs)
{
  if (lhs.idx().size() || rhs.idx().size())
    throw invalid_argument("subscript read must be performed on const variables");

  Ciphertext dest{};
  Compiler::active_func()->operate_binary(ir::OpCode::add, lhs, rhs, dest);
  return dest;
}

Plaintext operator+(const Plaintext &lhs, const Plaintext &rhs)
{
  if (lhs.idx().size() || rhs.idx().size())
    throw invalid_argument("subscript read must be performed on const variables");

  Plaintext dest{};
  Compiler::active_func()->operate_binary(ir::OpCode::add, lhs, rhs, dest);
  return dest;
}

Plaintext operator+(const Plaintext &lhs, const Scalar &rhs)
{
  if (lhs.idx().size())
    throw invalid_argument("subscript read must be performed on const variables");

  Plaintext dest{};
  Compiler::active_func()->operate_binary(ir::OpCode::add, lhs, rhs, dest);
  return dest;
}

Ciphertext operator+(const Scalar &lhs, const Ciphertext &rhs)
{
  if (rhs.idx().size())
    throw invalid_argument("subscript read must be performed on const variables");

  Ciphertext dest{};
  Compiler::active_func()->operate_binary(ir::OpCode::add, lhs, rhs, dest);
  return dest;
}

Plaintext operator+(const Scalar &lhs, const Plaintext &rhs)
{
  if (rhs.idx().size())
    throw invalid_argument("subscript read must be performed on const variables");

  Plaintext dest{};
  Compiler::active_func()->operate_binary(ir::OpCode::add, lhs, rhs, dest);
  return dest;
}

Scalar operator+(const Scalar &lhs, const Scalar &rhs)
{
  Scalar dest{};
  Compiler::active_func()->operate_binary(ir::OpCode::add, lhs, rhs, dest);
  return dest;
}

// addition assignement
Ciphertext &operator+=(Ciphertext &lhs, const Ciphertext &rhs)
{
  lhs = lhs + rhs;
  return lhs;
}

Ciphertext &operator+=(Ciphertext &lhs, const Plaintext &rhs)
{
  lhs = lhs + rhs;
  return lhs;
}

Ciphertext &operator+=(Ciphertext &lhs, const Scalar &rhs)
{
  lhs = lhs + rhs;
  return lhs;
}

Plaintext &operator+=(Plaintext &lhs, const Plaintext &rhs)
{
  lhs = lhs + rhs;
  return lhs;
}

Plaintext &operator+=(Plaintext &lhs, const Scalar &rhs)
{
  lhs = lhs + rhs;
  return lhs;
}

Scalar &operator+=(Scalar &lhs, const Scalar &rhs)
{
  lhs = lhs + rhs;
  return lhs;
}

// subtraction
Ciphertext operator-(const Ciphertext &lhs, const Ciphertext &rhs)
{
  if (lhs.idx().size() || rhs.idx().size())
    throw invalid_argument("subscript read must be performed on const variables");

  Ciphertext dest{};
  Compiler::active_func()->operate_binary(ir::OpCode::sub, lhs, rhs, dest);
  return dest;
}

Ciphertext operator-(const Ciphertext &lhs, const Plaintext &rhs)
{
  if (lhs.idx().size() || rhs.idx().size())
    throw invalid_argument("subscript read must be performed on const variables");

  Ciphertext dest{};
  Compiler::active_func()->operate_binary(ir::OpCode::sub, lhs, rhs, dest);
  return dest;
}

Ciphertext operator-(const Ciphertext &lhs, const Scalar &rhs)
{
  if (lhs.idx().size())
    throw invalid_argument("subscript read must be performed on const variables");

  Ciphertext dest{};
  Compiler::active_func()->operate_binary(ir::OpCode::sub, lhs, rhs, dest);
  return dest;
}

Ciphertext operator-(const Plaintext &lhs, const Ciphertext &rhs)
{
  if (lhs.idx().size() || rhs.idx().size())
    throw invalid_argument("subscript read must be performed on const variables");

  Ciphertext dest{};
  Compiler::active_func()->operate_binary(ir::OpCode::sub, lhs, rhs, dest);
  return dest;
}

Plaintext operator-(const Plaintext &lhs, const Plaintext &rhs)
{
  if (lhs.idx().size() || rhs.idx().size())
    throw invalid_argument("subscript read must be performed on const variables");

  Plaintext dest{};
  Compiler::active_func()->operate_binary(ir::OpCode::sub, lhs, rhs, dest);
  return dest;
}

Plaintext operator-(const Plaintext &lhs, const Scalar &rhs)
{
  if (lhs.idx().size())
    throw invalid_argument("subscript read must be performed on const variables");

  Plaintext dest{};
  Compiler::active_func()->operate_binary(ir::OpCode::sub, lhs, rhs, dest);
  return dest;
}

Ciphertext operator-(const Scalar &lhs, const Ciphertext &rhs)
{
  if (rhs.idx().size())
    throw invalid_argument("subscript read must be performed on const variables");

  Ciphertext dest{};
  Compiler::active_func()->operate_binary(ir::OpCode::sub, lhs, rhs, dest);
  return dest;
}

Plaintext operator-(const Scalar &lhs, const Plaintext &rhs)
{
  if (rhs.idx().size())
    throw invalid_argument("subscript read must be performed on const variables");

  Plaintext dest{};
  Compiler::active_func()->operate_binary(ir::OpCode::sub, lhs, rhs, dest);
  return dest;
}

Scalar operator-(const Scalar &lhs, const Scalar &rhs)
{
  Scalar dest{};
  Compiler::active_func()->operate_binary(ir::OpCode::sub, lhs, rhs, dest);
  return dest;
}

// subtraction assignement
Ciphertext &operator-=(Ciphertext &lhs, const Ciphertext &rhs)
{
  lhs = lhs - rhs;
  return lhs;
}

Ciphertext &operator-=(Ciphertext &lhs, const Plaintext &rhs)
{
  lhs = lhs - rhs;
  return lhs;
}

Ciphertext &operator-=(Ciphertext &lhs, const Scalar &rhs)
{
  lhs = lhs - rhs;
  return lhs;
}

Plaintext &operator-=(Plaintext &lhs, const Plaintext &rhs)
{
  lhs = lhs - rhs;
  return lhs;
}

Plaintext &operator-=(Plaintext &lhs, const Scalar &rhs)
{
  lhs = lhs - rhs;
  return lhs;
}

Scalar &operator-=(Scalar &lhs, const Scalar &rhs)
{
  lhs = lhs - rhs;
  return lhs;
}

// multiplication
Ciphertext operator*(const Ciphertext &lhs, const Ciphertext &rhs)
{
  if (lhs.idx().size() || rhs.idx().size())
    throw invalid_argument("subscript read must be performed on const variables");

  Ciphertext dest{};
  Compiler::active_func()->operate_binary(ir::OpCode::mul, lhs, rhs, dest);
  return dest;
}

Ciphertext operator*(const Ciphertext &lhs, const Plaintext &rhs)
{
  if (lhs.idx().size() || rhs.idx().size())
    throw invalid_argument("subscript read must be performed on const variables");

  Ciphertext dest{};
  Compiler::active_func()->operate_binary(ir::OpCode::mul, lhs, rhs, dest);
  return dest;
}

Ciphertext operator*(const Ciphertext &lhs, const Scalar &rhs)
{
  if (lhs.idx().size())
    throw invalid_argument("subscript read must be performed on const variables");

  Ciphertext dest{};
  Compiler::active_func()->operate_binary(ir::OpCode::mul, lhs, rhs, dest);
  return dest;
}

Ciphertext operator*(const Plaintext &lhs, const Ciphertext &rhs)
{
  if (lhs.idx().size() || rhs.idx().size())
    throw invalid_argument("subscript read must be performed on const variables");

  Ciphertext dest{};
  Compiler::active_func()->operate_binary(ir::OpCode::mul, lhs, rhs, dest);
  return dest;
}

Plaintext operator*(const Plaintext &lhs, const Plaintext &rhs)
{
  if (lhs.idx().size() || rhs.idx().size())
    throw invalid_argument("subscript read must be performed on const variables");

  Plaintext dest{};
  Compiler::active_func()->operate_binary(ir::OpCode::mul, lhs, rhs, dest);
  return dest;
}

Plaintext operator*(const Plaintext &lhs, const Scalar &rhs)
{
  if (lhs.idx().size())
    throw invalid_argument("subscript read must be performed on const variables");

  Plaintext dest{};
  Compiler::active_func()->operate_binary(ir::OpCode::mul, lhs, rhs, dest);
  return dest;
}

Ciphertext operator*(const Scalar &lhs, const Ciphertext &rhs)
{
  if (rhs.idx().size())
    throw invalid_argument("subscript read must be performed on const variables");

  Ciphertext dest{};
  Compiler::active_func()->operate_binary(ir::OpCode::mul, lhs, rhs, dest);
  return dest;
}

Plaintext operator*(const Scalar &lhs, const Plaintext &rhs)
{
  if (rhs.idx().size())
    throw invalid_argument("subscript read must be performed on const variables");

  Plaintext dest{};
  Compiler::active_func()->operate_binary(ir::OpCode::mul, lhs, rhs, dest);
  return dest;
}

Scalar operator*(const Scalar &lhs, const Scalar &rhs)
{
  Scalar dest{};
  Compiler::active_func()->operate_binary(ir::OpCode::mul, lhs, rhs, dest);
  return dest;
}

// multiplication assignement
Ciphertext &operator*=(Ciphertext &lhs, const Ciphertext &rhs)
{
  lhs = lhs * rhs;
  return lhs;
}

Ciphertext &operator*=(Ciphertext &lhs, const Plaintext &rhs)
{
  lhs = lhs * rhs;
  return lhs;
}

Ciphertext &operator*=(Ciphertext &lhs, const Scalar &rhs)
{
  lhs = lhs * rhs;
  return lhs;
}

Plaintext &operator*=(Plaintext &lhs, const Plaintext &rhs)
{
  lhs = lhs * rhs;
  return lhs;
}

Plaintext &operator*=(Plaintext &lhs, const Scalar &rhs)
{
  lhs = lhs * rhs;
  return lhs;
}

Scalar &operator*=(Scalar &lhs, const Scalar &rhs)
{
  lhs = lhs * rhs;
  return lhs;
}

// negation
Ciphertext operator-(const Ciphertext &arg)
{
  if (arg.idx().size())
    throw invalid_argument("subscript read must be performed on const variables");

  Ciphertext dest{};
  Compiler::active_func()->operate_unary(ir::OpCode::negate, arg, dest);
  return dest;
}

Plaintext operator-(const Plaintext &arg)
{
  if (arg.idx().size())
    throw invalid_argument("subscript read must be performed on const variables");

  Plaintext dest{};
  Compiler::active_func()->operate_unary(ir::OpCode::negate, arg, dest);
  return dest;
}

Scalar operator-(const Scalar &arg)
{
  Scalar dest{};
  Compiler::active_func()->operate_unary(ir::OpCode::negate, arg, dest);
  return dest;
}

// rotation
Ciphertext rotate(const Ciphertext &arg, int steps)
{
  if (arg.idx().size())
    throw invalid_argument("subscript read must be performed on const variables");

  // this should work since slot_count is a power of 2
  steps = static_cast<int>(steps % Compiler::active_func()->clear_data_evaluator().slot_count());
  Ciphertext dest{};
  Compiler::active_func()->operate_unary(ir::OpCode::rotate(steps), arg, dest);
  return dest;
}

Plaintext rotate(const Plaintext &arg, int steps)
{
  if (arg.idx().size())
    throw invalid_argument("subscript read must be performed on const variables");

  // this should work since slot_count is a power of 2
  steps = static_cast<int>(steps % Compiler::active_func()->clear_data_evaluator().slot_count());
  Plaintext dest{};
  Compiler::active_func()->operate_unary(ir::OpCode::rotate(steps), arg, dest);
  return dest;
}

Ciphertext operator<<(const Ciphertext &arg, int steps)
{
  return rotate(arg, steps);
}

Ciphertext operator>>(const Ciphertext &arg, int steps)
{
  return rotate(arg, -steps);
}

Plaintext operator<<(const Plaintext &arg, int steps)
{
  return rotate(arg, steps);
}

Plaintext operator>>(const Plaintext &arg, int steps)
{
  return rotate(arg, -steps);
}

// rotation assignement
Ciphertext &operator<<=(Ciphertext &arg, int steps)
{
  arg = arg << steps;
  return arg;
}

Ciphertext &operator>>=(Ciphertext &arg, int steps)
{
  arg = arg >> steps;
  return arg;
}

Plaintext &operator<<=(Plaintext &arg, int steps)
{
  arg = arg << steps;
  return arg;
}

Plaintext &operator>>=(Plaintext &arg, int steps)
{
  arg = arg >> steps;
  return arg;
}

// encryption
Ciphertext encrypt(const Plaintext &arg)
{
  if (arg.idx().size())
    throw invalid_argument("subscript read must be performed on const variables");

  Ciphertext dest{};
  Compiler::active_func()->operate_unary(ir::OpCode::encrypt, arg, dest);
  return dest;
}

Ciphertext encrypt(const Scalar &arg)
{
  Ciphertext dest{};
  Compiler::active_func()->operate_unary(ir::OpCode::encrypt, arg, dest);
  return dest;
}

// subscript helper
pair<int, int> compute_subscripted_range(const Ciphertext &arg)
{
  if (arg.idx().empty())
    throw invalid_argument("subscripted with empty index");

  int dim = arg.dim();
  const auto &shape = Compiler::active_func()->shape();
  size_t start = 0;
  for (auto axis_idx : arg.idx())
  {
    size_t lower_dims_slot_count = 1;
    for (size_t i = shape.size() - dim + 1; i < shape.size(); ++i)
      lower_dims_slot_count *= shape[i];
    start += axis_idx * lower_dims_slot_count;
    --dim;
  }
  size_t end = start + ((dim > 0) ? shape[shape.size() - dim] : 1);
  return {start, end};
}

pair<int, int> compute_subscripted_range(const Plaintext &arg)
{
  if (arg.idx().empty())
    throw invalid_argument("subscripted with empty index");

  int dim = arg.dim();
  const auto &shape = Compiler::active_func()->shape();
  size_t start = 0;
  for (auto axis_idx : arg.idx())
  {
    size_t lower_dims_slot_count = 1;
    for (size_t i = shape.size() - dim + 1; i < shape.size(); ++i)
      lower_dims_slot_count *= shape[i];
    start += axis_idx * lower_dims_slot_count;
    --dim;
  }
  size_t end = start + ((dim > 0) ? shape[shape.size() - dim] : 1);
  return {start, end};
}

// subscripted_read
Ciphertext emulate_subscripted_read(const Ciphertext &arg)
{
  auto [start, end] = compute_subscripted_range(arg);
  PackedVal mask(Compiler::active_func()->clear_data_evaluator().slot_count(), 0);

  for (size_t i = start; i < end; ++i)
    mask[i] = 1;
  Ciphertext resolved = arg;
  resolved.idx_.clear();
  Plaintext plain_mask(mask);
  plain_mask.dim_ = arg.dim();
  Ciphertext result = (resolved * plain_mask) << start;
  result.dim_ -= arg.idx_.size();
  return result;
}

Plaintext emulate_subscripted_read(const Plaintext &arg)
{
  auto [start, end] = compute_subscripted_range(arg);
  PackedVal mask(Compiler::active_func()->clear_data_evaluator().slot_count(), 0);
  for (size_t i = start; i < end; ++i)
    mask[i] = 1;
  Plaintext resolved = arg;
  resolved.idx_.clear();
  Plaintext plain_mask(mask);
  plain_mask.dim_ = arg.dim();
  Plaintext result = (resolved * plain_mask) << start;
  result.dim_ -= arg.idx_.size();
  return result;
}

void emulate_subscripted_write(Ciphertext &lhs, const Ciphertext &rhs)
{
  if (rhs.idx().size())
    throw invalid_argument("subscript read must be performed on const variables");

  auto [start, end] = compute_subscripted_range(lhs);
  size_t actual_dim = lhs.dim_;
  lhs.dim_ -= lhs.idx_.size();
  lhs.idx_.clear();
  if (Compiler::active_func()->is_valid_term_id(lhs.id()))
  {
    PackedVal mask(Compiler::active_func()->clear_data_evaluator().slot_count(), 1);
    for (size_t i = start; i < end; ++i)
      mask[i] = 0;
    Plaintext plain_mask(mask);
    plain_mask.dim_ = lhs.dim();
    lhs = lhs * plain_mask + (rhs >> start);
    lhs.dim_ = actual_dim;
  }
  else
  {
    lhs = rhs >> start;
    lhs.dim_ = actual_dim;
  }
}

void emulate_subscripted_write(Plaintext &lhs, const Plaintext &rhs)
{
  if (rhs.idx().size())
    throw invalid_argument("subscript read must be performed on const variables");

  auto [start, end] = compute_subscripted_range(lhs);
  size_t actual_dim = lhs.dim_;
  lhs.dim_ -= lhs.idx_.size();
  lhs.idx_.clear();
  if (Compiler::active_func()->is_valid_term_id(lhs.id()))
  {
    PackedVal mask(Compiler::active_func()->clear_data_evaluator().slot_count(), 1);
    for (size_t i = start; i < end; ++i)
      mask[i] = 0;
    Plaintext plain_mask(mask);
    plain_mask.dim_ = lhs.dim();
    lhs = lhs * plain_mask + (rhs >> start);
    lhs.dim_ = actual_dim;
  }
  else
  {
    lhs = rhs >> start;
    lhs.dim_ = actual_dim;
  }
}

// square
Ciphertext square(const Ciphertext &arg)
{
  Ciphertext dest{};
  Compiler::active_func()->operate_unary(ir::OpCode::square, arg, dest);
  return dest;
}

Plaintext square(const Plaintext &arg)
{
  return arg * arg;
}

// add_many
Ciphertext add_many(const vector<Ciphertext> &args)
{
  if (args.empty())
    return encrypt(Scalar(0));

  vector<Ciphertext> sum_vec;
  for (size_t i = 0; i < args.size() - 1; i += 2)
    sum_vec.push_back(args[i] + args[i + 1]);

  if (args.size() & 1)
    sum_vec.push_back(args.back());

  for (size_t i = 0; i < sum_vec.size() - 1; i += 2)
    sum_vec.push_back(sum_vec[i] + sum_vec[i + 1]);

  return sum_vec.back();
}

Plaintext add_many(const vector<Plaintext> &args)
{
  Plaintext result(PackedVal{0});
  for (size_t i = 0; i < args.size(); ++i)
    result += args[i];
  return result;
}

// mul_many
Ciphertext mul_many(const vector<Ciphertext> &args)
{
  if (args.empty())
    return encrypt(Scalar(0));

  vector<Ciphertext> sum_vec;
  for (size_t i = 0; i < args.size() - 1; i += 2)
    sum_vec.push_back(args[i] * args[i + 1]);

  if (args.size() & 1)
    sum_vec.push_back(args.back());

  for (size_t i = 0; i < sum_vec.size() - 1; i += 2)
    sum_vec.push_back(sum_vec[i] * sum_vec[i + 1]);

  return sum_vec.back();
}

Plaintext mul_many(const vector<Plaintext> &args)
{
  Plaintext result(1);
  for (size_t i = 0; i < args.size(); ++i)
    result *= args[i];
  return result;
}

// exponentiate
Ciphertext exponentiate(const Ciphertext &arg, std::uint64_t exponent)
{
  vector<Ciphertext> args(static_cast<size_t>(exponent), arg);
  return mul_many(args);
}
Plaintext exponentiate(const Plaintext &arg, std::uint64_t exponent)
{
  vector<Plaintext> args(static_cast<size_t>(exponent), arg);
  return mul_many(args);
}

// reduce_add
Ciphertext reduce_add(const Ciphertext &arg)
{
  Ciphertext result = arg;
  size_t steps = Compiler::active_func()->clear_data_evaluator().slot_count() >> 1;
  while (steps > 0)
  {
    result += result << steps;
    steps >>= 1;
  }
  return result;
}

Plaintext reduce_add(const Plaintext &arg)
{
  Plaintext result(0);
  for (size_t i = 0; i < Compiler::active_func()->clear_data_evaluator().slot_count(); ++i)
    result += arg << i;
  return result;
}

// reduce_mul
Ciphertext reduce_mul(const Ciphertext &arg)
{
  Ciphertext result = arg;
  size_t steps = Compiler::active_func()->clear_data_evaluator().slot_count() >> 1;
  while (steps > 0)
  {
    result *= result << steps;
    steps >>= 1;
  }
  return result;
}

Plaintext reduce_mul(const Plaintext &arg)
{
  Plaintext result(1);
  for (size_t i = 0; i < Compiler::active_func()->clear_data_evaluator().slot_count(); ++i)
    result *= arg << i;
  return result;
}
} // namespace fheco
