#pragma once

#include "seal/seal.h"
#include "ufhe/api/encryptor.hpp"
#include "ufhe/seal_backend/ciphertext.hpp"
#include "ufhe/seal_backend/encryption_context.hpp"
#include "ufhe/seal_backend/plaintext.hpp"
#include "ufhe/seal_backend/public_key.hpp"
#include "ufhe/seal_backend/secret_key.hpp"
#include <memory>

namespace ufhe
{
namespace seal_backend
{
  class Encryptor : public api::Encryptor
  {
  public:
    Encryptor(const EncryptionContext &context, const PublicKey &public_key)
      : underlying_(std::make_shared<seal::Encryptor>(context.underlying(), public_key.underlying()))
    {}

    Encryptor(const EncryptionContext &context, const SecretKey &secret_key)
      : underlying_(std::make_shared<seal::Encryptor>(context.underlying(), secret_key.underlying()))
    {}

    Encryptor(const EncryptionContext &context, const PublicKey &public_key, const SecretKey &secret_key)
      : underlying_(
          std::make_shared<seal::Encryptor>(context.underlying(), public_key.underlying(), secret_key.underlying()))
    {}

    inline api::backend_type backend() const override { return api::backend_type::seal; }

    inline void encrypt(const api::Plaintext &plain, api::Ciphertext &destination) const override
    {
      underlying().encrypt(
        dynamic_cast<const Plaintext &>(plain).underlying(), *dynamic_cast<Ciphertext &>(destination).underlying_);
    }

    inline void encrypt_symmetric(const api::Plaintext &plain, api::Ciphertext &destination) const override
    {
      underlying().encrypt_symmetric(
        dynamic_cast<const Plaintext &>(plain).underlying(), *dynamic_cast<Ciphertext &>(destination).underlying_);
    }

    inline const seal::Encryptor &underlying() const { return *underlying_; }

  private:
    std::shared_ptr<seal::Encryptor> underlying_;
  };
} // namespace seal_backend
} // namespace ufhe