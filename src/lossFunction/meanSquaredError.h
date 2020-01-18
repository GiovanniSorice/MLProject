//
// Created by checco on 01/01/20.
//

#ifndef MLPROJECT_SRC_LOSSFUNCTION_MEANSQUAREDERROR_H_
#define MLPROJECT_SRC_LOSSFUNCTION_MEANSQUAREDERROR_H_

#include "lossFunction.h"
class MeanSquaredError : public LossFunction {
 public:
  void Error(const arma::mat &&trainLabelsBatch,
             arma::mat &&outputActivateBatch,
             arma::mat &&currentError) override;
  void ComputePartialDerivative(const arma::mat &&trainLabelsBatch,
                                arma::mat &&outputActivateBatch,
                                arma::mat &&partialDerivativeOutput) override;
  ~MeanSquaredError() override = default;
};

#endif //MLPROJECT_SRC_LOSSFUNCTION_MEANSQUAREDERROR_H_
