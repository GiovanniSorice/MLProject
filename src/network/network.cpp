//
// Created by gs1010 on 29/11/19.
//

#include "network.h"
#include "../preprocessing/preprocessing.h"


Network::Network(Preprocessing &dataPreprocessor) : preprocessor(dataPreprocessor) {}

void Network::Add(Layer &layer) {
  net.push_back(layer);
}

/**
 *  Initilialize the weight parameter of the layer stored inside the network
 *
 *  @param upperBound The maximum random value generated
 *  @param lowerBound The minimum random value generated
 * */
void Network::Init(const double upperBound = 1, const double lowerBound = -1) {
  for (Layer &i : net) {
    i.Init(upperBound, lowerBound);
  }
}

//! std::move is used to do a cheap move and not do a deep copy of arma::mat training set
void Network::Train(int trainPercent, int batchSizePercent) {
  arma::mat trainingSet;
  preprocessor.GetTrainingSet(60, 20, 20, std::move(trainingSet));

  // TODO: split the training set in batch
  forward(std::move(trainingSet));
}

/**
 *  Iterate over the raw in the batch and pass them to the network
 * */
void Network::forward(arma::mat &&batch) {

  for (int i = 0; i < batch.n_rows; ++i) {
    const arma::mat
        currentRaw = batch.row(i); // TODO: .row(i) is doing deep copy .unsafe_col(i) doesn't but return column
    std::cout << "Row " << i << ":" << currentRaw << std::endl;
    arma::mat outputWeight;
    arma::mat activateMatrix;
    for (Layer &i : net) {
      i.Forward(std::move(currentRaw), std::move(outputWeight));
      i.Activate(outputWeight, std::move(activateMatrix));
      activateMatrix.impl_print("Actual activate Matrix");
    }
  }
}