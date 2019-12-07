//
// Created by checco on 02/12/19.
//

#ifndef MLPROJECT_SRC_ACTIVATIONFUNCTION_ACTIVATIONFUNCTION_H_
#define MLPROJECT_SRC_ACTIVATIONFUNCTION_ACTIVATIONFUNCTION_H_

#include "armadillo"

class ActivationFunction {
 public:
  virtual double Compute(const double x) = 0;
  virtual void Derive(const arma::mat &input, arma::mat &output) = 0;
};

#endif //MLPROJECT_SRC_ACTIVATIONFUNCTION_ACTIVATIONFUNCTION_H_
