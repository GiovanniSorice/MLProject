//
// Created by checco on 01/01/20.
//

#ifndef MLPROJECT_SRC_LOSSFUNCTION_LOSSFUNCTION_H_
#define MLPROJECT_SRC_LOSSFUNCTION_LOSSFUNCTION_H_

#include "armadillo"

class LossFunction {
 public:
  virtual void Error(const arma::mat &&trainLabelsBatch,
                     arma::mat &&outputActivateBatch,
                     arma::mat &&currentError) = 0;
  virtual void ComputePartialDerivative(const arma::mat &&trainLabelsBatch,
                                        arma::mat &&outputActivateBatch,
                                        arma::mat &&errorBatch) = 0;
  virtual ~LossFunction() = default;
};
#endif //MLPROJECT_SRC_LOSSFUNCTION_LOSSFUNCTION_H_
