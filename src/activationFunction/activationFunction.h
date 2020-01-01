//
// Created by checco on 02/12/19.
//

#ifndef MLPROJECT_SRC_ACTIVATIONFUNCTION_ACTIVATIONFUNCTION_H_
#define MLPROJECT_SRC_ACTIVATIONFUNCTION_ACTIVATIONFUNCTION_H_

#include "armadillo"

class ActivationFunction {
 public:
  virtual void Compute(const arma::mat &input, arma::mat &&output) = 0;
  virtual double Compute(const double x) = 0;
  virtual void Derive(const arma::mat &&input, arma::mat &&output) = 0;
  virtual double Derive(const double x) = 0;
  virtual ~ActivationFunction() = default;
};

#endif //MLPROJECT_SRC_ACTIVATIONFUNCTION_ACTIVATIONFUNCTION_H_
