//
// Created by gs1010 on 19/01/20.
//

#ifndef MLPROJECT_SRC_ACTIVATIONFUNCTION_LINEARFUNCTION_H_
#define MLPROJECT_SRC_ACTIVATIONFUNCTION_LINEARFUNCTION_H_

#include "activationFunction.h"
class LinearFunction : public ActivationFunction {
 public:
  double Compute(const double x) override;
  void Compute(const arma::mat &input, arma::mat &&output) override;
  void Derive(const arma::mat &&input, arma::mat &&output) override;
  double Derive(const double x) override;
  ~LinearFunction() override = default;

};

#endif //MLPROJECT_SRC_ACTIVATIONFUNCTION_LINEARFUNCTION_H_
