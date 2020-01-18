//
// Created by checco on 01/01/20.
//

#include "binaryCrossentropy.h"

void BinaryCrossentropy::Error(const arma::mat &&trainLabelsBatch, arma::mat &&outputActivateBatch) {
  arma::mat errorBatch = -arma::mean(
      trainLabelsBatch % arma::log(outputActivateBatch) + (1 - trainLabelsBatch) % arma::log(1 - outputActivateBatch));

  //outputActivateBatch.print("outputActivateBatch");
  //trainLabelsBatch.print("trainLabelsBatch");
  //(arma::log(outputActivateBatch)).print("arma::log outputActivate batch");
  //(trainLabelsBatch % arma::log(outputActivateBatch)).print("trainLabelsBatch % arma::log(outputActivateBatch)");
  //errorBatch.print("Current net error");

}

void BinaryCrossentropy::ComputePartialDerivative(const arma::mat &&trainLabelsBatch,
                                                  arma::mat &&outputActivateBatch,
                                                  arma::mat &&partialDerivativeOutput) {
  partialDerivativeOutput = arma::sum(
      (-trainLabelsBatch / outputActivateBatch) + (1 - trainLabelsBatch) / (1 - outputActivateBatch));
  //partialDerivativeOutput.print("Partial derivative output");
  //(-trainLabelsBatch / outputActivateBatch).print("(-trainLabelsBatch / outputActivateBatch)");
  //((1 - trainLabelsBatch) / (1 - outputActivateBatch)).print("(1 - trainLabelsBatch) / (1 - outputActivateBatch)");
  //((-trainLabelsBatch / outputActivateBatch) + (1 - trainLabelsBatch) / (1 - outputActivateBatch)).print(
  //  "(-trainLabelsBatch / outputActivateBatch) + (1 - trainLabelsBatch) / (1 - outputActivateBatch))");

}