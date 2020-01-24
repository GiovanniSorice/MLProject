//
// Created by checco on 24/01/20.
//

#include "meanEuclideanError.h"
void MeanEuclideanError::ComputePartialDerivative(const arma::mat &&trainLabelsBatch,
                                                  arma::mat &&outputActivateBatch,
                                                  arma::mat &&errorBatch) {

}

void MeanEuclideanError::Error(const arma::mat &&trainLabelsBatch,
                               arma::mat &&outputActivateBatch,
                               arma::mat &&currentError) {
  currentError = arma::mean(arma::sqrt(arma::sum(arma::pow(outputActivateBatch - trainLabelsBatch, 2))));
}
