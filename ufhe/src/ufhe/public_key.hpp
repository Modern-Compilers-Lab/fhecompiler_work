#pragma once

#include "ufhe/api/public_key.hpp"
#include "ufhe/config.hpp"
#include "ufhe/seal_backend/public_key.hpp"
#include <memory>

namespace ufhe
{
class PublicKey : public api::PublicKey
{
  friend class KeyGenerator;

public:
  PublicKey();

  inline api::backend_type backend() const override { return underlying().backend(); }

  inline const api::PublicKey &underlying() const { return *underlying_; }

private:
  std::shared_ptr<api::PublicKey> underlying_;
};
} // namespace ufhe