//
// Created by gs1010 on 07/12/19.
//

#ifndef MLPROJECT_SRC_ACTIVATIONFUNCTION_TANHFUNCTION_H_
#define MLPROJECT_SRC_ACTIVATIONFUNCTION_TANHFUNCTION_H_
#include "activationFunction.h"
class TanhFunction : public ActivationFunction {
  virtual double Compute(const double x) override;
  virtual void Compute(const arma::mat &input, arma::mat &&output) override;
  virtual void Derive(const arma::mat &&input, arma::mat &&output) override;
 public:
  ~TanhFunction() override = default;
};

#endif //MLPROJECT_SRC_ACTIVATIONFUNCTION_TANHFUNCTION_H_
