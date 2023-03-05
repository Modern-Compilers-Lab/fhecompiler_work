#pragma once
#include "trs.hpp"
#include "trs_util_functions.hpp"

#define CIRCUIT_SAVE_FLAG true

namespace fheco_trs
{

namespace ruleset
{

  using T = MatchingTerm;

  T x(TermType::variable);
  /*
    T x(TermType::ciphertextType);
  */
  T y(TermType::variable);
  /*
    T y(TermType::plaintextType);
  */
  T z(TermType::variable);
  T k(TermType::variable);
  T n(TermType::constant);
  T m(TermType::constant);
  T p(TermType::rawDataType);
  T q(TermType::rawDataType);

  /*
    std::vector<RewriteRule> dummy_ruleset = {
      {(x << p), x, p == 0},
      {(x << p) << q, x << T::fold(p + q)},
      {(x + (y << p)) + (z << q), x + (((y << T::fold(p - q)) + z) << q),
       (p > q) && (p > 0) && (q > 0) && T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate),
    CIRCUIT_SAVE_FLAG},
      {((y << p) + x) + (z << q), x + (((y << T::fold(p - q)) + z) << q),
       (p > q) && (p > 0) && (q > 0) && T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate),
    CIRCUIT_SAVE_FLAG},
      {(z << q) + (x + (y << p)), x + (((y << T::fold(p - q)) + z) << q),
       (p > q) && (p > 0) && (q > 0) && T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate),
    CIRCUIT_SAVE_FLAG},
      {(z << q) + ((y << p) + x), x + (((y << T::fold(p - q)) + z) << q),
       (p > q) && (p > 0) && (q > 0) && T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate),
    CIRCUIT_SAVE_FLAG},
      {(x + (y << p)) + (z << q), x + (((z << T::fold(p - q)) + y) << q),
       (p < q) && (p > 0) && (q > 0) && T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate),
    CIRCUIT_SAVE_FLAG},
      {((y << p) + x) + (z << q), x + (((z << T::fold(q - p)) + y) << q),
       (p < q) && (p > 0) && (q > 0) && T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate),
    CIRCUIT_SAVE_FLAG},
      {(z << q) + (x + (y << p)), x + (((z << T::fold(q - p)) + y) << q),
       (p < q) && (p > 0) && (q > 0) && T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate),
    CIRCUIT_SAVE_FLAG},
      {(z << q) + ((y << p) + x), x + (((z << T::fold(q - p)) + y) << q),
       (p < q) && (p > 0) && (q > 0) && T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate),
    CIRCUIT_SAVE_FLAG},
      {(x + (y << p)) + (z << q), x + (((y << T::fold(p - q)) + z) << q),
       (p < q) && (p < 0) && (q < 0) && T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate),
    CIRCUIT_SAVE_FLAG},
      {((y << p) + x) + (z << q), x + (((y << T::fold(p - q)) + z) << q),
       (p < q) && (p < 0) && (q < 0) && T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate),
    CIRCUIT_SAVE_FLAG},
      {(z << q) + (x + (y << p)), x + (((y << T::fold(p - q)) + z) << q),
       (p < q) && (p < 0) && (q < 0) && T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate),
    CIRCUIT_SAVE_FLAG},
      {(z << q) + ((y << p) + x), x + (((y << T::fold(p - q)) + z) << q),
       (p < q) && (p < 0) && (q < 0) && T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate),
    CIRCUIT_SAVE_FLAG},
      {(x + (y << p)) + (z << p), x + ((y + z) << p), T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate)},
      {((y << p) + x) + (z << p), x + ((y + z) << p), T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate)},
      {(z << p) + (x + (y << p)), x + ((z + y) << p), T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate)},
      {(z << p) + ((y << p) + x), x + ((z + y) << p), T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate)},
      {(x << p) + (y << p), (x + y) << p},
      {(x << p) * (y << p), (x * y) << p},
      {(x << p) - (y << p), (x - y) << p},
      {(x << p) + (y << q), ((x << T::fold(p - q)) + y) << q, (p > q) && (p > 0) && (q > 0), CIRCUIT_SAVE_FLAG},
      {(x << p) + (y << q), ((y << T::fold(q - p)) + x) << p, (q > p) && (p > 0) && (q > 0), CIRCUIT_SAVE_FLAG},
      {(x << p) + (y << q), ((x << T::fold(p - q)) + y) << q, (p < q) && (p > 0) && (q > 0), CIRCUIT_SAVE_FLAG},
      {(x << p) + (y << q), ((y << T::fold(q - p)) + x) << p, (q < p) && (p > 0) && (q > 0), CIRCUIT_SAVE_FLAG},
      {(x << p) + (y << q), ((x << T::fold(p - q)) + y) << q, (p < q) && (p < 0) && (q < 0), CIRCUIT_SAVE_FLAG},
      {(x << p) + (y << q), ((y << T::fold(q - p)) + x) << p, (q < p) && (p < 0) && (q < 0), CIRCUIT_SAVE_FLAG},
      {(x << p) + (y << q), ((x << T::fold(p - q)) + y) << q, (p < q) && (p < 0) && (q < 0), CIRCUIT_SAVE_FLAG},
      {(x << p) + (y << q), ((y << T::fold(q - p)) + x) << p, (q < p) && (p < 0) && (q < 0), CIRCUIT_SAVE_FLAG}};
    */

  std::vector<RewriteRule> rotations_ruleset = {{(x << p), x, p == 0}, {(x << p) << q, x << T::fold(p + q)}};

  T c0(TermType::constant);
  T c1(TermType::constant);
  T t1(TermType::ciphertextType);
  T t2(TermType::ciphertextType);
  T t3(TermType::ciphertextType);
  T t4(TermType::ciphertextType);

  std::vector<RewriteRule> mul_ruleset = {
    {(t1 * (t2 * (t3 * t4))), ((t1 * t2) * (t3 * t4))},
    {x * c0, c0, T::iszero(c0)} /*rule1*/,
    {c0 * x, c0, T::iszero(c0)} /*rule1*/,
    {x * c0, x, T::isone(c0)} /*rule2*/,
    {c0 * x, x, T::isone(c0)} /*rule2*/,
    {(x + c0) * (x + c1), (x * (x + T::fold(c0 + c1)) + T::fold(c0 * c1))} /*rule3*/,
    {(x + c0) * (c1 + x), (x * (x + T::fold(c0 + c1)) + T::fold(c0 * c1))} /*rule3*/,
    {(c0 + x) * (x + c1), (x * (x + T::fold(c0 + c1)) + T::fold(c0 * c1))} /*rule3*/,
    {(c0 + x) * (c1 + x), (x * (x + T::fold(c0 + c1)) + T::fold(c0 * c1))} /*rule3*/,
    {((x + c0) * c1), (x * c1 + T::fold(c0 * c1))} /*rule4*/,
    {((c0 + x) * c1), (x * c1 + T::fold(c0 * c1))} /*rule4*/,
    {(c1 * (x + c0)), (x * c1 + T::fold(c0 * c1))} /*rule4*/,
    {((c0 + x) * c1), (x * c1 + T::fold(c0 * c1))} /*rule4*/,
    {(c0 - x) * y, c0 - x *y, T::iszero(c0)} /*rule5*/,
    {y * (c0 - x), c0 - x *y, T::iszero(c0)} /*rule5*/,
    {((x * c0) * c1), (x * T::fold(c0 * c1))} /*rule6*/,
    {((c0 * x) * c1), (x * T::fold(c0 * c1))} /*rule6*/,
    {(c1 * (c0 * x)), (x * T::fold(c0 * c1))} /*rule6*/,
    {(c1 * (c0 * x)), (x * T::fold(c0 * c1))} /*rule6*/,
    {((x * c0) * y), ((x * y) * c0)} /*rule7*/,
    {((c0 * x) * y), ((x * y) * c0)} /*rule7*/,
    {(y * (x * c0)), ((x * y) * c0)} /*rule7*/,
    {(y * (c0 * x)), ((x * y) * c0)} /*rule7*/,
    {(x * x), square(x), MatchingTerm::type_of(x) == static_cast<int>(ir::TermType::ciphertextType)},
    {(x * y) * x, y *square(x), MatchingTerm::type_of(x) == static_cast<int>(ir::TermType::ciphertextType)},
    {x * (x * y), y *square(x), MatchingTerm::type_of(x) == static_cast<int>(ir::TermType::ciphertextType)},
    {(y * x) * x, y *square(x), MatchingTerm::type_of(x) == static_cast<int>(ir::TermType::ciphertextType)},
    {x * (y * x), y *square(x), MatchingTerm::type_of(x) == static_cast<int>(ir::TermType::ciphertextType)},
    {((x * y) * z) * x, (y * z) * square(x),
     MatchingTerm::type_of(x) == static_cast<int>(ir::TermType::ciphertextType)},
    {x * ((x * y) * z), (y * z) * square(x),
     MatchingTerm::type_of(x) == static_cast<int>(ir::TermType::ciphertextType)},
    {(x * (y << p)) * (z << q), x *(((y << T::fold(p - q)) * z) << q),
     (p > q) && (p > 0) && (q > 0) && T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate), CIRCUIT_SAVE_FLAG},
    {((y << p) * x) * (z << q), x *(((y << T::fold(p - q)) * z) << q),
     (p > q) && (p > 0) && (q > 0) && T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate), CIRCUIT_SAVE_FLAG},
    {(z << q) * (x * (y << p)), x *(((y << T::fold(p - q)) * z) << q),
     (p > q) && (p > 0) && (q > 0) && T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate), CIRCUIT_SAVE_FLAG},
    {(z << q) * ((y << p) * x), x *(((y << T::fold(p - q)) * z) << q),
     (p > q) && (p > 0) && (q > 0) && T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate), CIRCUIT_SAVE_FLAG},
    {(x * (y << p)) * (z << q), x *(((z << T::fold(p - q)) * y) << q),
     (p < q) && (p > 0) && (q > 0) && T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate), CIRCUIT_SAVE_FLAG},
    {((y << p) * x) * (z << q), x *(((z << T::fold(q - p)) * y) << q),
     (p < q) && (p > 0) && (q > 0) && T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate), CIRCUIT_SAVE_FLAG},
    {(z << q) * (x * (y << p)), x *(((z << T::fold(q - p)) * y) << q),
     (p < q) && (p > 0) && (q > 0) && T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate), CIRCUIT_SAVE_FLAG},
    {(z << q) * ((y << p) * x), x *(((z << T::fold(q - p)) * y) << q),
     (p < q) && (p > 0) && (q > 0) && T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate), CIRCUIT_SAVE_FLAG},
    {(x * (y << p)) * (z << q), x *(((y << T::fold(p - q)) * z) << q),
     (p < q) && (p < 0) && (q < 0) && T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate), CIRCUIT_SAVE_FLAG},
    {((y << p) * x) * (z << q), x *(((y << T::fold(p - q)) * z) << q),
     (p < q) && (p < 0) && (q < 0) && T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate), CIRCUIT_SAVE_FLAG},
    {(z << q) * (x * (y << p)), x *(((y << T::fold(p - q)) * z) << q),
     (p < q) && (p < 0) && (q < 0) && T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate), CIRCUIT_SAVE_FLAG},
    {(z << q) * ((y << p) * x), x *(((y << T::fold(p - q)) * z) << q),
     (p < q) && (p < 0) && (q < 0) && T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate), CIRCUIT_SAVE_FLAG},
    {(x * (y << p)) * (z << p), x *((y + z) << p), T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate)},
    {((y << p) * x) * (z << p), x *((y * z) << p), T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate)},
    {(z << p) * (x * (y << p)), x *((z * y) << p), T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate)},
    {(z << p) * ((y << p) * x), x *((z + y) << p), T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate)},
    {(x << p) * (y << p), (x * y) << p},
    {(x << p) * (y << q), ((x << T::fold(p - q)) * y) << q, (p > q) && (p > 0) && (q > 0), CIRCUIT_SAVE_FLAG},
    {(x << p) * (y << q), ((y << T::fold(q - p)) * x) << p, (q > p) && (p > 0) && (q > 0), CIRCUIT_SAVE_FLAG},
    {(x << p) * (y << q), ((x << T::fold(p - q)) * y) << q, (p < q) && (p > 0) && (q > 0), CIRCUIT_SAVE_FLAG},
    {(x << p) * (y << q), ((y << T::fold(q - p)) * x) << p, (q < p) && (p > 0) && (q > 0), CIRCUIT_SAVE_FLAG},
    {(x << p) * (y << q), ((x << T::fold(p - q)) * y) << q, (p < q) && (p < 0) && (q < 0), CIRCUIT_SAVE_FLAG},
    {(x << p) * (y << q), ((y << T::fold(q - p)) * x) << p, (q < p) && (p < 0) && (q < 0), CIRCUIT_SAVE_FLAG},
    {(x << p) * (y << q), ((x << T::fold(p - q)) * y) << q, (p < q) && (p < 0) && (q < 0), CIRCUIT_SAVE_FLAG},
    {(x << p) * (y << q), ((y << T::fold(q - p)) * x) << p, (q < p) && (p < 0) && (q < 0), CIRCUIT_SAVE_FLAG}};

  T w(TermType::variable);
  T u(TermType::variable);

  std::vector<RewriteRule> sub_ruleset = {
    {x - c0, x, T::iszero(c0)} /*rule1*/,
    {c0 - c1, T::fold(c0 - c1)} /*I think we don't need this*/ /*rule2*/,
    {x - x, 0} /*rule 3*/,
    {(x + y) - x, y} /*rule4*/,
    {x - (x + y), 0 - y} /*not from halide*/ /*rule5*/,
    {(x + c0) - c1, x + T::fold(c0 - c1)} /*rule6*/,
    {(x + y) - (x + z), y - z} /*rule7*/,
    {((x + y) + z) - x, y + z} /*rule8*/,
    {x - (y + (x - z)), z - y} /*rule9*/,
    {x - (y + (x + z)), 0 - (y + z)} /*rule10*/,
    {(x + y) - (z + (w + x)), y - (z + w)} /*rule11*/,
    {(x + y) - (z + (x + w)), y - (z + w)} /*rule11*/,
    {(x + y) - (x + (z + w)), y - (z + w)} /*rule11*/,
    {(y + x) - (z + (w + x)), y - (z + w)} /*rule11*/,
    {(y + x) - (z + (x + w)), y - (z + w)} /*rule11*/,
    {(y + x) - (x + (z + w)), y - (z + w)} /*rule11*/,
    {(x - y) - (x + z), (0 - y - z)} /*rule12*/,
    {(x - y) - (z + x), (0 - y - z)} /*rule13*/,
    {(((x + y) - z) - x), (y - z)} /*rule14*/,
    {(((y + x) - z) - x), (y - z)} /*rule14*/,
    {(0 - (x + (y - z))), (z - (x + y))} /*rule15*/,
    {(0 - ((y - z) + x)), (z - (x + y))} /*rule15*/,
    {(((x - y) - z) - x), (0 - (y + z))} /*rule16*/,
    {(x * y - z * y), ((x - z) * y)} /*rule17*/,
    {(y * x - z * y), ((x - z) * y)} /*rule17*/,
    {(x * y - y * z), ((x - z) * y)} /*rule17*/,
    {(y * x - y * z), ((x - z) * y)} /*rule17*/,
    {((u + x * y) - z * y), (u + (x - z) * y)} /*rule18*/,
    {((x * y + u) - z * y), (u + (x - z) * y)} /*rule18*/,
    {((u + y * x) - z * y), (u + (x - z) * y)} /*rule18*/,
    {((y * x + u) - z * y), (u + (x - z) * y)} /*rule18*/,
    {((u + x * y) - y * z), (u + (x - z) * y)} /*rule18*/,
    {((x * y + u) - y * z), (u + (x - z) * y)} /*rule18*/,
    {((u + y * x) - y * z), (u + (x - z) * y)} /*rule18*/,
    {((y * x + u) - y * z), (u + (x - z) * y)} /*rule18*/,
    {(x * y - x), (x * (y - 1))} /*rule19*/,
    {(y * x - x), (x * (y - 1))} /*rule19*/,
    {((y << p) - x) - (z << q), (((y << T::fold(p - q)) - z) << q) - x,
     (p > q) && (p > 0) && (q > 0) && T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate), CIRCUIT_SAVE_FLAG},
    {(z << q) - ((y << p) - x), (((y << T::fold(p - q)) - z) << q) - x,
     (p > q) && (p > 0) && (q > 0) && T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate), CIRCUIT_SAVE_FLAG},
    {((y << p) - x) - (z << q), (((z << T::fold(p - q)) - y) << q) - x,
     (p < q) && (p > 0) && (q > 0) && T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate), CIRCUIT_SAVE_FLAG},
    {((y << p) - x) - (z << p), ((y - z) << p) - x, T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate)},
    {(z << p) - ((y << p) - x), ((z - y) << p) - x, T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate)},
    {(x << p) - (y << p), (x - y) << p},
    {(x << p) - (y << q), ((x << T::fold(p - q)) - y) << q, (p > q) && (p > 0) && (q > 0), CIRCUIT_SAVE_FLAG},
    {(x << p) - (y << q), ((y << T::fold(q - p)) - x) << p, (q > p) && (p > 0) && (q > 0), CIRCUIT_SAVE_FLAG},
    {(x << p) - (y << q), ((x << T::fold(p - q)) - y) << q, (p < q) && (p > 0) && (q > 0), CIRCUIT_SAVE_FLAG},
    {(x << p) - (y << q), ((y << T::fold(q - p)) - x) << p, (q < p) && (p > 0) && (q > 0), CIRCUIT_SAVE_FLAG},
    {(x << p) - (y << q), ((x << T::fold(p - q)) - y) << q, (p < q) && (p < 0) && (q < 0), CIRCUIT_SAVE_FLAG},
    {(x << p) - (y << q), ((y << T::fold(q - p)) - x) << p, (q < p) && (p < 0) && (q < 0), CIRCUIT_SAVE_FLAG},
    {(x << p) - (y << q), ((x << T::fold(p - q)) - y) << q, (p < q) && (p < 0) && (q < 0), CIRCUIT_SAVE_FLAG},
    {(x << p) - (y << q), ((y << T::fold(q - p)) - x) << p, (q < p) && (p < 0) && (q < 0), CIRCUIT_SAVE_FLAG}};

  std::vector<RewriteRule> add_ruleset = {
    {x + c0, x, T::iszero(c0)} /*rule1*/,
    {c0 + x, x, T::iszero(c0)} /*rule2*/,
    {(x + y * (-1)), (x - y)} /*rule3*/,
    {(y * (-1) + x), (x - y)} /*rule3*/,
    {(x + (-1) * y), (x - y)} /*rule3*/,
    {((-1) * y + x), (x - y)} /*rule3*/,
    {((x + c0) + c1), (x + T::fold(c0 + c1))} /*rule4*/,
    {((c0 + x) + c1), (x + T::fold(c0 + c1))} /*rule4*/,
    {(c1 + (x + c0)), (x + T::fold(c0 + c1))} /*rule4*/,
    {(c1 + (c0 + x)), (x + T::fold(c0 + c1))} /*rule4*/,
    {((x + c0) + y), ((x + y) + c0)} /*this might be removed later*/ /*rule5*/,
    {((c0 + x) + y), ((x + y) + c0)} /*this might be removed later*/ /*rule5*/,
    {(y + (x + c0)), ((x + y) + c0)} /*this might be removed later*/ /*rule5*/,
    {(y + (c0 + x)), ((x + y) + c0)} /*this might be removed later*/ /*rule5*/,
    {((c0 - x) + c1), (T::fold(c0 + c1) - x)} /*rule6*/,
    {(c1 + (c0 - x)), (T::fold(c0 + c1) - x)} /*rule6*/,
    {((c0 - x) + y), ((y - x) + c0)} /*rule7*/,
    {(y + (c0 - x)), ((y - x) + c0)} /*rule7*/,
    {((x - y) + y), x} /*rule8*/,
    {(y + (x - y)), x} /*rule8*/,
    {(((x - y) + z) + y), (x + z)} /*rule9*/,
    {((z + (x - y)) + y), (x + z)} /*rule9*/,
    {(y + ((x - y) + z)), (x + z)} /*rule9*/,
    {(y + (z + (x - y))), (x + z)} /*rule9*/,
    {(x + (c0 - y)), ((x - y) + c0)} /*rule10*/,
    {((x - y) + (y - z)), (x - z)} /*rule11*/,
    {((y - z) + (x - y)), (x - z)} /*rule11*/,
    {((x - y) + (y + z)), (x + z)} /*rule12*/,
    {((x - y) + (z + y)), (x + z)} /*rule12*/,
    {((y + z) + (x - y)), (x + z)} /*rule12*/,
    {((z + y) + (x - y)), (x + z)} /*rule12*/,
    {(x + ((y - x) - z)), (y - z)} /*rule13*/,
    {(((y - x) - z) + x), (y - z)} /*rule13*/,
    {(x + ((0 - y) - z)), (x - (y + z))} /*rule14*/,
    {(((0 - y) - z) + x), (x - (y + z))} /*rule14*/,
    {(((c0 - x) - y) + c1), (T::fold(c0 + c1) - y) - x} /*rule15*/,
    {(c1 + ((c0 - x) - y)), (T::fold(c0 + c1) - y) - x} /*rule15*/,
    {(x * y + z * y), ((x + z) * y)} /*rule16*/,
    {(y * x + z * y), ((x + z) * y)} /*rule16*/,
    {(x * y + y * z), ((x + z) * y)} /*rule16*/,
    {(y * x + y * z), ((x + z) * y)} /*rule16*/,
    {((x * y) + (z * y)), ((x + z) * y)} /*rule16*/,
    {((y * x) + (z * y)), ((x + z) * y)} /*rule16*/,
    {((x * y) + (y * z)), ((x + z) * y)} /*rule16*/,
    {((y * x) + (y * z)), ((x + z) * y)} /*rule16*/,
    {(x + x * y), (x * (y + 1))} /*rule17*/,
    {(x * y + x), (x * (y + 1))} /*rule17*/,
    {(x + y * x), (x * (y + 1))} /*rule17*/,
    {(y * x + x), (x * (y + 1))} /*rule17*/,
    {(x + (y << p)) + (z << q), x + (((y << T::fold(p - q)) + z) << q),
     (p > q) && (p > 0) && (q > 0) && T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate), CIRCUIT_SAVE_FLAG},
    {((y << p) + x) + (z << q), x + (((y << T::fold(p - q)) + z) << q),
     (p > q) && (p > 0) && (q > 0) && T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate), CIRCUIT_SAVE_FLAG},
    {(z << q) + (x + (y << p)), x + (((y << T::fold(p - q)) + z) << q),
     (p > q) && (p > 0) && (q > 0) && T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate), CIRCUIT_SAVE_FLAG},
    {(z << q) + ((y << p) + x), x + (((y << T::fold(p - q)) + z) << q),
     (p > q) && (p > 0) && (q > 0) && T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate), CIRCUIT_SAVE_FLAG},
    {(x + (y << p)) + (z << q), x + (((z << T::fold(p - q)) + y) << q),
     (p < q) && (p > 0) && (q > 0) && T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate), CIRCUIT_SAVE_FLAG},
    {((y << p) + x) + (z << q), x + (((z << T::fold(q - p)) + y) << q),
     (p < q) && (p > 0) && (q > 0) && T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate), CIRCUIT_SAVE_FLAG},
    {(z << q) + (x + (y << p)), x + (((z << T::fold(q - p)) + y) << q),
     (p < q) && (p > 0) && (q > 0) && T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate), CIRCUIT_SAVE_FLAG},
    {(z << q) + ((y << p) + x), x + (((z << T::fold(q - p)) + y) << q),
     (p < q) && (p > 0) && (q > 0) && T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate), CIRCUIT_SAVE_FLAG},
    {(x + (y << p)) + (z << q), x + (((y << T::fold(p - q)) + z) << q),
     (p < q) && (p < 0) && (q < 0) && T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate), CIRCUIT_SAVE_FLAG},
    {((y << p) + x) + (z << q), x + (((y << T::fold(p - q)) + z) << q),
     (p < q) && (p < 0) && (q < 0) && T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate), CIRCUIT_SAVE_FLAG},
    {(z << q) + (x + (y << p)), x + (((y << T::fold(p - q)) + z) << q),
     (p < q) && (p < 0) && (q < 0) && T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate), CIRCUIT_SAVE_FLAG},
    {(z << q) + ((y << p) + x), x + (((y << T::fold(p - q)) + z) << q),
     (p < q) && (p < 0) && (q < 0) && T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate), CIRCUIT_SAVE_FLAG},
    {(x + (y << p)) + (z << p), x + ((y + z) << p), T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate)},
    {((y << p) + x) + (z << p), x + ((y + z) << p), T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate)},
    {(z << p) + (x + (y << p)), x + ((z + y) << p), T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate)},
    {(z << p) + ((y << p) + x), x + ((z + y) << p), T::opcode_of(x) != static_cast<int>(ir::OpCode::rotate)},
    {(x << p) + (y << p), (x + y) << p},
    {(x << p) + (y << q), ((x << T::fold(p - q)) + y) << q, (p > q) && (p > 0) && (q > 0), CIRCUIT_SAVE_FLAG},
    {(x << p) + (y << q), ((y << T::fold(q - p)) + x) << p, (q > p) && (p > 0) && (q > 0), CIRCUIT_SAVE_FLAG},
    {(x << p) + (y << q), ((x << T::fold(p - q)) + y) << q, (p < q) && (p > 0) && (q > 0), CIRCUIT_SAVE_FLAG},
    {(x << p) + (y << q), ((y << T::fold(q - p)) + x) << p, (q < p) && (p > 0) && (q > 0), CIRCUIT_SAVE_FLAG},
    {(x << p) + (y << q), ((x << T::fold(p - q)) + y) << q, (p < q) && (p < 0) && (q < 0), CIRCUIT_SAVE_FLAG},
    {(x << p) + (y << q), ((y << T::fold(q - p)) + x) << p, (q < p) && (p < 0) && (q < 0), CIRCUIT_SAVE_FLAG},
    {(x << p) + (y << q), ((x << T::fold(p - q)) + y) << q, (p < q) && (p < 0) && (q < 0), CIRCUIT_SAVE_FLAG},
    {(x << p) + (y << q), ((y << T::fold(q - p)) + x) << p, (q < p) && (p < 0) && (q < 0), CIRCUIT_SAVE_FLAG}};

} // namespace ruleset

} // namespace fheco_trs
