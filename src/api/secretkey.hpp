#pragma once

namespace api
{
class SecretKey
{
public:
  SecretKey() { init(); }

  // TODO: Allow access to the key parms_id

  // TODO: Serialization support
private:
  virtual void init() = 0;
};
} // namespace api
