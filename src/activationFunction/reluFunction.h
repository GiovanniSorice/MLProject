//
// Created by checco on 01/01/20.
//

#ifndef MLPROJECT_SRC_ACTIVATIONFUNCTION_RELUFUNCTION_H_
#define MLPROJECT_SRC_ACTIVATIONFUNCTION_RELUFUNCTION_H_

#include "activationFunction.h"

class ReluFunction : public ActivationFunction {
 public:
  double Compute(const double x) override;
  void Compute(const arma::mat &input, arma::mat &&output) override;
  void Derive(const arma::mat &&input, arma::mat &&output) override;
  double Derive(const double x) override;
  ~ReluFunction() override = default;
};

#endif //MLPROJECT_SRC_ACTIVATIONFUNCTION_RELUFUNCTION_H_
