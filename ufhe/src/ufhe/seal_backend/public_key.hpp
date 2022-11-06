#pragma once

#include "seal/seal.h"
#include "ufhe/api/public_key.hpp"
#include <memory>

namespace ufhe
{
namespace seal_backend
{
  class PublicKey : public api::PublicKey
  {
    friend class KeyGenerator;

  public:
    PublicKey() : underlying_(std::make_shared<seal::PublicKey>()) {}

    inline api::backend_type backend() const override { return api::backend_type::seal; }

    inline const seal::PublicKey &underlying() const { return *underlying_; }

  private:
    std::shared_ptr<seal::PublicKey> underlying_;
  };
} // namespace seal_backend
} // namespace ufhe