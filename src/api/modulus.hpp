#pragma once

#include <cstdint>

namespace api
{
class Modulus
{
public:
  Modulus(std::uint64_t value) { init(value); }

  virtual int bit_count() const = 0;

  virtual std::uint64_t value() const = 0;

  virtual bool is_prime() const = 0;

  virtual bool operator==(const Modulus &compare) const = 0;

  virtual bool operator!=(const Modulus &compare) const = 0;

  virtual bool operator<(const Modulus &compare) const = 0;

  virtual std::uint64_t reduce(std::uint64_t value) const = 0;

  // TODO: Serialization support

private:
  virtual void init(std::uint64_t value) = 0;
};
} // namespace api
