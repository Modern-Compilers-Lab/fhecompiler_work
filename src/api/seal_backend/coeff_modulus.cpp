#include "coeff_modulus.hpp"

namespace ufhe
{
namespace seal_backend
{
  CoeffModulus::CoeffModulus(const std::vector<Modulus> &moduli) : moduli_(moduli)
  {
    for (const Modulus &modulus : moduli)
      underlying_.push_back(modulus.underlying_);
  }

  CoeffModulus::CoeffModulus(std::vector<seal::Modulus> seal_moduli) : underlying_(seal_moduli), moduli_()
  {
    moduli_.reserve(underlying_.size());
    for (const seal::Modulus &seal_modulus : underlying_)
      moduli_.push_back(Modulus(seal_modulus));
  }

  IModulus::vector CoeffModulus::value() const
  {
    IModulus::vector moduli_wrappers(moduli_.begin(), moduli_.end());
    return moduli_wrappers;
  }
} // namespace seal_backend
} // namespace ufhe
