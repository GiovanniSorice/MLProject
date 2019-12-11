//
// Created by gs1010 on 09/12/19.
//

#ifndef MLPROJECT_SRC_ACTIVATIONFUNCTION_LOGISTICFUNCTION_H_
#define MLPROJECT_SRC_ACTIVATIONFUNCTION_LOGISTICFUNCTION_H_

#include "activationFunction.h"
class LogisticFunction : public ActivationFunction {
 public:
  virtual double Compute(const double x) override;
  virtual void Compute(const arma::mat &input, arma::mat &&output) override;
  virtual void Derive(const arma::mat &&input, arma::mat &&output) override;
  virtual double Derive(const double x) override;
  ~LogisticFunction() override = default;

};

#endif //MLPROJECT_SRC_ACTIVATIONFUNCTION_LOGISTICFUNCTION_H_
