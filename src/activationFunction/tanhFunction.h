//
// Created by gs1010 on 07/12/19.
//

#ifndef MLPROJECT_SRC_ACTIVATIONFUNCTION_TANHFUNCTION_H_
#define MLPROJECT_SRC_ACTIVATIONFUNCTION_TANHFUNCTION_H_
#include "activationFunction.h"
class tanhFunction : public ActivationFunction {
  double Compute(const double x) override;
  void Derive(const arma::mat &input, arma::mat &output) override;
};

#endif //MLPROJECT_SRC_ACTIVATIONFUNCTION_TANHFUNCTION_H_
