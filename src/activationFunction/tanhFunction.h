//
// Created by gs1010 on 07/12/19.
//

#ifndef MLPROJECT_SRC_ACTIVATIONFUNCTION_TANHFUNCTION_H_
#define MLPROJECT_SRC_ACTIVATIONFUNCTION_TANHFUNCTION_H_
#include "activationFunction.h"
class TanhFunction : public ActivationFunction {
 public:
  double Compute(const double x) override;
  void Compute(const arma::mat &input, arma::mat &&output) override;
  void Derive(const arma::mat &&input, arma::mat &&output) override;
  double Derive(const double x) override;
  ~TanhFunction() override = default;
};

#endif //MLPROJECT_SRC_ACTIVATIONFUNCTION_TANHFUNCTION_H_
