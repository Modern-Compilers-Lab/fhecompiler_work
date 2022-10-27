#pragma once

#include "ciphertext.hpp"
#include "encryption_parameters.hpp"
#include "fhecompiler_const.hpp"
#include "ops_overloads.hpp"
#include "params_selector.hpp"
#include "plaintext.hpp"
#include "program.hpp"
#include "scalar.hpp"
#include "translator.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

ir::Program *program;

namespace fhecompiler
{

void init(const std::string &program_name, size_t dim, Scheme program_scheme)
{
  static ir::Program program_object(program_name, dim);
  program = &program_object;
  program->set_scheme(program_scheme);
}

void compile(const std::string &output_filename)
{
  params_selector::ParameterSelector parameters_selector(program);
  params_selector::EncryptionParameters params = parameters_selector.select_parameters();

  translator::Translator tr(program, &params);

  {
    std::ofstream translation_os(output_filename);

    if (!translation_os)
      throw("couldn't open file for translation.\n");

    tr.translate(translation_os);

    translation_os.close();
  }
}

} // namespace fhecompiler
