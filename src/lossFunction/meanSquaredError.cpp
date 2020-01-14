//
// Created by checco on 01/01/20.
//

#include "meanSquaredError.h"

/**
 *  Compute the error of the network comparing the output produced with the desired
 *
 *  @param outputActivateBatch  Output value produced by the network
 *  @param trainLabelsBatch  Correct value of the data passed in the network
 * */
void MeanSquaredError::Error(const arma::mat &&trainLabelsBatch,
                             arma::mat &&outputActivateBatch) {
  trainLabelsBatch.print("TrainLabelsBatch");
  outputActivateBatch.print("OutputActivateBatch");

  arma::mat currentError = arma::mean(arma::pow(outputActivateBatch - trainLabelsBatch, 2));
  currentError.print("Net current error");

}

/** Compute and store in errorBatch the partial derivative of the output layer
 *
 * @param outputActivateBatch  Output value produced by the network
 * @param trainLabelsBatch  Correct value of the data passed in the network
 * @param errorBatch Partial derivative of output layer
 * */
void MeanSquaredError::ComputePartialDerivative(const arma::mat &&trainLabelsBatch,
                                                arma::mat &&outputActivateBatch,
                                                arma::mat &&errorBatch) {
  errorBatch = arma::mean(outputActivateBatch - trainLabelsBatch);
  errorBatch.print("errorBatch");
}
