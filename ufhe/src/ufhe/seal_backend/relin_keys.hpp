#pragma once

#include "seal/seal.h"
#include "ufhe/api/relin_keys.hpp"

namespace ufhe
{
namespace seal_backend
{
  class RelinKeys : public api::RelinKeys
  {
    friend class Evaluator;
    friend class KeyGenerator;

  public:
    RelinKeys() {}

    inline api::backend_type backend() const override { return api::backend_type::seal; }

    inline std::size_t size() const override { return underlying_.size(); }

  private:
    seal::RelinKeys underlying_;
  };
} // namespace seal_backend
} // namespace ufhe
