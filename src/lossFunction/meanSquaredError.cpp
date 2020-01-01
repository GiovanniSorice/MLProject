//
// Created by checco on 01/01/20.
//

#include "meanSquaredError.h"
void MeanSquaredError::Error(const arma::mat &&trainLabelsBatch,
                             arma::mat &&outputActivateBatch,
                             arma::mat &&errorBatch) {
  errorBatch = arma::mean(arma::pow(trainLabelsBatch - outputActivateBatch, 2));
}
