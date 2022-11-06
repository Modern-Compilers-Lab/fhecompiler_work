#pragma once

#include "ufhe/api/galois_keys.hpp"
#include "ufhe/config.hpp"
#include "ufhe/seal_backend/galois_keys.hpp"
#include <memory>

namespace ufhe
{
class GaloisKeys : public api::GaloisKeys
{
  friend class KeyGenerator;

public:
  GaloisKeys();

  inline api::backend_type backend() const override { return underlying().backend(); }

  inline std::size_t size() const override { return underlying().size(); }

  inline const api::GaloisKeys &underlying() const { return *underlying_; }

private:
  std::shared_ptr<api::GaloisKeys> underlying_;
};
} // namespace ufhe
