#include "fheco/trs/ops_overloads.hpp"
#include "fheco/trs/op_gen_op_code.hpp"
#include "fheco/trs/term_op_code.hpp"
#include <utility>

using namespace std;

namespace fheco::trs
{
// addition
TermMatcher operator+(TermMatcher lhs, TermMatcher rhs)
{
  return TermMatcher{TermOpCode::add, vector<TermMatcher>{move(lhs), move(rhs)}};
}

OpGenMatcher operator+(OpGenMatcher lhs, OpGenMatcher rhs)
{
  return OpGenMatcher{OpGenOpCode::add, vector<OpGenMatcher>{move(lhs), move(rhs)}};
}

// subtraction
TermMatcher operator-(TermMatcher lhs, TermMatcher rhs)
{
  return TermMatcher{TermOpCode::sub, vector<TermMatcher>{move(lhs), move(rhs)}};
}

OpGenMatcher operator-(OpGenMatcher lhs, OpGenMatcher rhs)
{
  return OpGenMatcher{OpGenOpCode::sub, vector<OpGenMatcher>{move(lhs), move(rhs)}};
}

// multiplication
TermMatcher operator*(TermMatcher lhs, TermMatcher rhs)
{
  return TermMatcher{TermOpCode::mul, vector<TermMatcher>{move(lhs), move(rhs)}};
}

// negation
TermMatcher operator-(TermMatcher arg)
{
  return TermMatcher{TermOpCode::negate, vector<TermMatcher>{move(arg)}};
}

OpGenMatcher operator-(OpGenMatcher arg)
{
  return OpGenMatcher{OpGenOpCode::negate, vector<OpGenMatcher>{move(arg)}};
}

// rotation
TermMatcher operator<<(TermMatcher arg, OpGenMatcher steps)
{
  return TermMatcher{TermOpCode::rotate(move(steps)), vector<TermMatcher>{move(arg)}};
}

TermMatcher operator>>(TermMatcher arg, OpGenMatcher steps)
{
  return TermMatcher{TermOpCode::rotate(-move(steps)), vector<TermMatcher>{move(arg)}};
}

// encryption
TermMatcher encrypt(TermMatcher arg)
{
  return TermMatcher{TermOpCode::encrypt, vector<TermMatcher>{move(arg)}};
}

// square
TermMatcher square(TermMatcher arg)
{
  return TermMatcher{TermOpCode::square, vector<TermMatcher>{move(arg)}};
}
} // namespace fheco::trs
