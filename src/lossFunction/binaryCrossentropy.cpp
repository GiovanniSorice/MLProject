//
// Created by checco on 01/01/20.
//

#include "binaryCrossentropy.h"

void BinaryCrossentropy::Error(const arma::mat &&trainLabelsBatch,
                               arma::mat &&outputActivateBatch,
                               arma::mat &&currentError) {
  currentError = arma::mean(
      -trainLabelsBatch % arma::log(outputActivateBatch) - (1 - trainLabelsBatch) % arma::log(1 - outputActivateBatch));
}

void BinaryCrossentropy::ComputePartialDerivative(const arma::mat &&trainLabelsBatch,
                                                  arma::mat &&outputActivateBatch,
                                                  arma::mat &&partialDerivativeOutput) {
  partialDerivativeOutput = arma::sum(
      (-trainLabelsBatch / outputActivateBatch) + (1 - trainLabelsBatch) / (1 - outputActivateBatch));

}


