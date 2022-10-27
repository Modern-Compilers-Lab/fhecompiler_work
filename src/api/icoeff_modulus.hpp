#pragma once

#include "imodulus.hpp"
#include "interface.hpp"

namespace ufhe
{
class ICoeffModulus : public Interface
{
public:
  virtual IModulus::vector value() const = 0;

  // TODO: Serialization support
};
} // namespace ufhe
