#pragma once

#include <string>
#include <variant>

namespace datatype

{

typedef std::string rawData;
const std::string ct_label_prefix = "ciphertext";
const std::string sc_label_prefix = "scalar";
const std::string pt_label_prefix = "plaintext";
const std::string output_tag = "output";
inline const char *eval_not_supported = "operation/evaluatio not supported during compile time";

} // namespace datatype
