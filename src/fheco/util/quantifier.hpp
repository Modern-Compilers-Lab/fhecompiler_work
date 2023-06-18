#pragma once

#include "fheco/param_select/encryption_params.hpp"
#include "fheco/ir/common.hpp"
#include "fheco/ir/op_code.hpp"
#include "fheco/ir/term.hpp"
#include <cstddef>
#include <memory>
#include <ostream>
#include <unordered_map>
#include <utility>

namespace fheco::ir
{
class Func;
} // namespace fheco::ir

namespace fheco::util
{
class Quantifier
{
public:
  struct DepthSummary
  {
    double min_xdepth_;
    double min_depth_;
    double avg_xdepth_;
    double avg_depth_;
    double max_xdepth_;
    double max_depth_;
  };

  struct CtxtTermDepthInfo
  {
    double xdepth_;
    double depth_;
  };

  using CtxtTermsDepthInfo =
    std::unordered_map<const ir::Term *, CtxtTermDepthInfo, ir::Term::HashPtr, ir::Term::EqualPtr>;

  // ctxt ctxt operation info
  struct CCOpInfo
  {
    double opposite_level_;
    double arg1_size_;
    double arg2_size_;
  };
  struct HashCCOpInfo
  {
    std::size_t operator()(const CCOpInfo &cc_op) const;
  };
  struct EqualCCOpInfo
  {
    bool operator()(const CCOpInfo &lhs, const CCOpInfo &rhs) const;
  };
  using CCOpCount = std::unordered_map<CCOpInfo, double, HashCCOpInfo, EqualCCOpInfo>;

  // ctxt any operation info
  struct CAOpInfo
  {
    double opposite_level_;
    double arg_size_;
  };
  struct HashCAOpInfo
  {
    std::size_t operator()(const CAOpInfo &ca_op_info) const;
  };
  struct EqualCAOpInfo
  {
    bool operator()(const CAOpInfo &lhs, const CAOpInfo &rhs) const;
  };
  using CAOpCount = std::unordered_map<CAOpInfo, double, HashCAOpInfo, EqualCAOpInfo>;

  struct CtxtTermInfo
  {
    double opposite_level_;
    double size_;
  };

  using CtxtTermsInfo = std::unordered_map<const ir::Term *, CtxtTermInfo, ir::Term::HashPtr, ir::Term::EqualPtr>;

  Quantifier(std::shared_ptr<ir::Func> func) : func_{std::move(func)} {}

  void run_all_analysis(const param_select::EncryptionParams &params);

  void run_all_analysis();

  void compute_he_depth_info();

  void count_terms_classes();

  void compute_global_metrics(const param_select::EncryptionParams &params);

  void print_info(std::ostream &os, bool depth_details = false);

  void print_he_depth_info(std::ostream &os, bool details) const;

  void print_terms_classes_info(std::ostream &os) const;

  void print_global_metrics(std::ostream &os) const;

  inline const std::shared_ptr<ir::Func> func() const { return func_; }

  inline bool depth_metrics() const { return depth_metrics_; }

  inline const DepthSummary &he_depth_summary() const { return he_depth_summary_; }

  inline const CtxtTermsDepthInfo &ctxt_leaves_depth_info() const { return ctxt_leaves_depth_info_; }

  inline bool terms_classes_metrics() const { return terms_classes_metrics_; }

  inline double relin_keys_count() const { return relin_keys_count_; }

  inline double rotation_keys_count() const { return rotation_keys_count_; }

  inline double all_terms_count() const { return all_terms_count_; }

  inline double captured_terms_count() const { return captured_terms_count_; }

  inline double ptxt_leaves_count() const { return ptxt_leaves_count_; }

  inline double pp_ops_count() const { return pp_ops_count_; }

  inline double ctxt_leaves_count() const { return ctxt_leaves_count_; }

  inline const CCOpCount &cc_mul_count() const { return cc_mul_count_; }

  inline const CAOpCount &he_square_count() const { return he_square_count_; }

  inline double encrypt_count() const { return encrypt_count_; }

  inline const CAOpCount &relin_count() const { return relin_count_; }

  inline const CAOpCount &he_rotate_count() const { return he_rotate_count_; }

  inline const CAOpCount &cp_mul_count() const { return cp_mul_count_; }

  inline const CAOpCount &mod_switch_count() const { return mod_switch_count_; }

  inline const CAOpCount &he_add_sub_negate_count() const { return he_add_sub_negate_count_; }

  inline const CtxtTermsInfo &ctxt_output_terms_info() const { return ctxt_output_terms_info_; }

  inline double circuit_static_cost() const { return circuit_static_cost_; }

  inline bool global_metrics() const { return global_metrics_; }

  inline double circuit_cost() const { return circuit_cost_; }

  inline double rotation_keys_total_size() const { return rotation_keys_total_size_; }

  inline double relin_keys_total_size() const { return relin_keys_total_size_; }

  inline double ctxt_input_terms_total_size() const { return ctxt_input_terms_total_size_; }

  inline double ctxt_input_terms_count() const { return ctxt_input_terms_count_; }

  inline double ctxt_output_terms_total_size() const { return ctxt_output_terms_total_size_; }

private:
  std::shared_ptr<ir::Func> func_;

  bool depth_metrics_ = false;

  DepthSummary he_depth_summary_{};

  CtxtTermsDepthInfo ctxt_leaves_depth_info_{};

  bool terms_classes_metrics_ = false;

  double relin_keys_count_ = 0;

  double rotation_keys_count_ = 0;

  double all_terms_count_ = 0;

  double captured_terms_count_ = 0;

  double ptxt_leaves_count_ = 0;

  double pp_ops_count_ = 0;

  double ctxt_leaves_count_ = 0;

  CCOpCount cc_mul_count_{};

  CAOpCount he_square_count_{};

  double encrypt_count_ = 0;

  CAOpCount relin_count_{};

  CAOpCount he_rotate_count_{};

  CAOpCount cp_mul_count_{};

  CAOpCount mod_switch_count_{};

  CAOpCount he_add_sub_negate_count_{};

  CtxtTermsInfo ctxt_output_terms_info_{};

  double circuit_static_cost_ = 0;

  bool global_metrics_ = false;

  double circuit_cost_ = 0;

  double rotation_keys_total_size_ = 0;

  double relin_keys_total_size_ = 0;

  double ctxt_input_terms_total_size_ = 0;

  double ctxt_input_terms_count_ = 0;

  double ctxt_output_terms_total_size_ = 0;

  friend Quantifier operator/(const Quantifier &lhs, const Quantifier &rhs);
  friend Quantifier operator-(const Quantifier &lhs, const Quantifier &rhs);
  friend Quantifier operator*(const Quantifier &lhs, int coeff);
};

Quantifier operator/(const Quantifier &lhs, const Quantifier &rhs);

Quantifier &operator/=(Quantifier &lhs, const Quantifier &rhs);

Quantifier operator-(const Quantifier &lhs, const Quantifier &rhs);

Quantifier &operator-=(Quantifier &lhs, const Quantifier &rhs);

Quantifier operator*(const Quantifier &lhs, int coeff);

Quantifier operator*(int coeff, const Quantifier &rhs);

Quantifier operator*=(Quantifier &lhs, int coeff);

Quantifier::DepthSummary operator/(const Quantifier::DepthSummary &lhs, const Quantifier::DepthSummary &rhs);

Quantifier::DepthSummary &operator/=(Quantifier::DepthSummary &lhs, const Quantifier::DepthSummary &rhs);

Quantifier::DepthSummary operator-(const Quantifier::DepthSummary &lhs, const Quantifier::DepthSummary &rhs);

Quantifier::DepthSummary &operator-=(Quantifier::DepthSummary &lhs, const Quantifier::DepthSummary &rhs);

Quantifier::DepthSummary operator*(const Quantifier::DepthSummary &lhs, int coeff);

Quantifier::DepthSummary operator*(int coeff, const Quantifier::DepthSummary &rhs);

Quantifier::DepthSummary operator*=(Quantifier::DepthSummary &lhs, int coeff);

Quantifier::CtxtTermsDepthInfo operator/(
  const Quantifier::CtxtTermsDepthInfo &lhs, const Quantifier::CtxtTermsDepthInfo &rhs);

Quantifier::CtxtTermsDepthInfo &operator/=(
  Quantifier::CtxtTermsDepthInfo &lhs, const Quantifier::CtxtTermsDepthInfo &rhs);

Quantifier::CtxtTermsDepthInfo operator-(
  const Quantifier::CtxtTermsDepthInfo &lhs, const Quantifier::CtxtTermsDepthInfo &rhs);

Quantifier::CtxtTermsDepthInfo &operator-=(
  Quantifier::CtxtTermsDepthInfo &lhs, const Quantifier::CtxtTermsDepthInfo &rhs);

Quantifier::CtxtTermsDepthInfo operator*(const Quantifier::CtxtTermsDepthInfo &lhs, int coeff);

Quantifier::CtxtTermsDepthInfo operator*(int coeff, const Quantifier::CtxtTermsDepthInfo &rhs);

Quantifier::CtxtTermsDepthInfo operator*=(Quantifier::CtxtTermsDepthInfo &lhs, int coeff);

Quantifier::CtxtTermDepthInfo operator/(
  const Quantifier::CtxtTermDepthInfo &lhs, const Quantifier::CtxtTermDepthInfo &rhs);

Quantifier::CtxtTermDepthInfo &operator/=(Quantifier::CtxtTermDepthInfo &lhs, const Quantifier::CtxtTermDepthInfo &rhs);

Quantifier::CtxtTermDepthInfo operator-(
  const Quantifier::CtxtTermDepthInfo &lhs, const Quantifier::CtxtTermDepthInfo &rhs);

Quantifier::CtxtTermDepthInfo &operator-=(Quantifier::CtxtTermDepthInfo &lhs, const Quantifier::CtxtTermDepthInfo &rhs);

Quantifier::CtxtTermDepthInfo operator*(const Quantifier::CtxtTermDepthInfo &lhs, int coeff);

Quantifier::CtxtTermDepthInfo operator*(int coeff, const Quantifier::CtxtTermDepthInfo &rhs);

Quantifier::CtxtTermDepthInfo operator*=(Quantifier::CtxtTermDepthInfo &lhs, int coeff);

Quantifier::CCOpCount operator/(const Quantifier::CCOpCount &lhs, const Quantifier::CCOpCount &rhs);

Quantifier::CCOpCount &operator/=(Quantifier::CCOpCount &lhs, const Quantifier::CCOpCount &rhs);

Quantifier::CCOpCount operator-(const Quantifier::CCOpCount &lhs, const Quantifier::CCOpCount &rhs);

Quantifier::CCOpCount &operator-=(Quantifier::CCOpCount &lhs, const Quantifier::CCOpCount &rhs);

Quantifier::CCOpCount operator*(const Quantifier::CCOpCount &lhs, int coeff);

Quantifier::CCOpCount operator*(int coeff, const Quantifier::CCOpCount &rhs);

Quantifier::CCOpCount operator*=(Quantifier::CCOpCount &lhs, int coeff);

Quantifier::CAOpCount operator/(const Quantifier::CAOpCount &lhs, const Quantifier::CAOpCount &rhs);

Quantifier::CAOpCount &operator/=(Quantifier::CAOpCount &lhs, const Quantifier::CAOpCount &rhs);

Quantifier::CAOpCount operator-(const Quantifier::CAOpCount &lhs, const Quantifier::CAOpCount &rhs);

Quantifier::CAOpCount &operator-=(Quantifier::CAOpCount &lhs, const Quantifier::CAOpCount &rhs);

Quantifier::CAOpCount operator*(const Quantifier::CAOpCount &lhs, int coeff);

Quantifier::CAOpCount operator*(int coeff, const Quantifier::CAOpCount &rhs);

Quantifier::CAOpCount operator*=(Quantifier::CAOpCount &lhs, int coeff);

Quantifier::CtxtTermsInfo operator/(const Quantifier::CtxtTermsInfo &lhs, const Quantifier::CtxtTermsInfo &rhs);

Quantifier::CtxtTermsInfo &operator/=(Quantifier::CtxtTermsInfo &lhs, const Quantifier::CtxtTermsInfo &rhs);

Quantifier::CtxtTermsInfo operator-(const Quantifier::CtxtTermsInfo &lhs, const Quantifier::CtxtTermsInfo &rhs);

Quantifier::CtxtTermsInfo &operator-=(Quantifier::CtxtTermsInfo &lhs, const Quantifier::CtxtTermsInfo &rhs);

Quantifier::CtxtTermsInfo operator*(const Quantifier::CtxtTermsInfo &lhs, int coeff);

Quantifier::CtxtTermsInfo operator*(int coeff, const Quantifier::CtxtTermsInfo &rhs);

Quantifier::CtxtTermsInfo operator*=(Quantifier::CtxtTermsInfo &lhs, int coeff);

Quantifier::CtxtTermInfo operator/(const Quantifier::CtxtTermInfo &lhs, const Quantifier::CtxtTermInfo &rhs);

Quantifier::CtxtTermInfo &operator/=(Quantifier::CtxtTermInfo &lhs, const Quantifier::CtxtTermInfo &rhs);

Quantifier::CtxtTermInfo operator-(const Quantifier::CtxtTermInfo &lhs, const Quantifier::CtxtTermInfo &rhs);

Quantifier::CtxtTermInfo &operator-=(Quantifier::CtxtTermInfo &lhs, const Quantifier::CtxtTermInfo &rhs);

Quantifier::CtxtTermInfo operator*(const Quantifier::CtxtTermInfo &lhs, int coeff);

Quantifier::CtxtTermInfo operator*(int coeff, const Quantifier::CtxtTermInfo &rhs);

Quantifier::CtxtTermInfo operator*=(Quantifier::CtxtTermInfo &lhs, int coeff);

std::ostream &operator<<(std::ostream &os, const Quantifier &quantifier);

std::ostream &operator<<(std::ostream &os, const Quantifier::CtxtTermsDepthInfo &ctxt_terms_depth_info);

std::ostream &operator<<(std::ostream &os, const Quantifier::CtxtTermDepthInfo &ctxt_term_depth_info);

std::ostream &operator<<(std::ostream &os, const Quantifier::CCOpCount &cc_op_count);

std::ostream &operator<<(std::ostream &os, const Quantifier::CAOpCount &ca_op_count);

std::ostream &operator<<(std::ostream &os, const Quantifier::CtxtTermsInfo &ctxt_terms_info);

std::ostream &operator<<(std::ostream &os, const Quantifier::CtxtTermInfo &ctxt_term_info);
} // namespace fheco::util
