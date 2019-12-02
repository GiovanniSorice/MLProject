//
// Created by checco on 02/12/19.
//

#ifndef MLPROJECT_SRC_ACTIVATIONFUNCTION_ACTIVATIONFUNCTION_H_
#define MLPROJECT_SRC_ACTIVATIONFUNCTION_ACTIVATIONFUNCTION_H_

#include "armadillo"

class activationFunction {
 public:
  virtual arma::mat input() = 0;
  virtual arma::mat output() = 0;
};

#endif //MLPROJECT_SRC_ACTIVATIONFUNCTION_ACTIVATIONFUNCTION_H_
