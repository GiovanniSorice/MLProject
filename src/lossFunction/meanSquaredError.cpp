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
  //trainLabelsBatch.raw_print(arma::cout, "TrainLabelsBatch");
  //outputActivateBatch.raw_print(arma::cout, "OutputActivateBatch");
  arma::mat currentError = arma::mean(arma::pow(trainLabelsBatch - outputActivateBatch, 2), 1) / 2;
  //TODO: controllare in caso di più di un output
  //currentError.raw_print(arma::cout, "Net current error");
}

/** Compute and store in errorBatch the partial derivative of the output layer
 *
 * @param outputActivateBatch  Output value produced by the network
 * @param trainLabelsBatch  Correct value of the data passed in the network
 * @param partialDerivativeOutput Partial derivative of output layer
 * */
void MeanSquaredError::ComputePartialDerivative(const arma::mat &&trainLabelsBatch,
                                                arma::mat &&outputActivateBatch,
                                                arma::mat &&partialDerivativeOutput) {
  //trainLabelsBatch.print("Desired output");
  //outputActivateBatch.print("Net  output");
  partialDerivativeOutput = arma::sum(trainLabelsBatch - outputActivateBatch, 1);
  //TODO: controllare in caso di più di un output
  //partialDerivateOutput.print("partial derivative output");
}
