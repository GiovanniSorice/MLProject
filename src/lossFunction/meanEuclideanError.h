//
// Created by checco on 24/01/20.
//

#ifndef MLPROJECT_SRC_LOSSFUNCTION_MEANEUCLIDEANERROR_H_
#define MLPROJECT_SRC_LOSSFUNCTION_MEANEUCLIDEANERROR_H_

#include "lossFunction.h"

class MeanEuclideanError : public LossFunction {
 public:
  void Error(const arma::mat &&trainLabelsBatch,
             arma::mat &&outputActivateBatch,
             arma::mat &&currentError) override;
  void ComputePartialDerivative(const arma::mat &&trainLabelsBatch,
                                arma::mat &&outputActivateBatch,
                                arma::mat &&errorBatch) override;
  ~MeanEuclideanError() override = default;
};

#endif //MLPROJECT_SRC_LOSSFUNCTION_MEANEUCLIDEANERROR_H_
