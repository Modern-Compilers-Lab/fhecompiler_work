#pragma once

#include "ufhe/api/scheme.hpp"
#include "ufhe/config.hpp"
#include "ufhe/seal_backend/scheme.hpp"
#include <memory>

namespace ufhe
{
class Scheme : public api::Scheme
{
public:
  Scheme(api::backend_type backend, api::scheme_type scheme);

  Scheme(api::scheme_type scheme): Scheme(Config::backend(), scheme){}

  inline api::backend_type backend() const override { return underlying().backend(); }

  inline api::implementation_level level() const override { return api::implementation_level::high_level; }

  inline api::scheme_type type() const override { return underlying().type(); }

  inline const api::Scheme &underlying() const { return *underlying_; }

private:
  std::shared_ptr<api::Scheme> underlying_;
};
} // namespace ufhe
