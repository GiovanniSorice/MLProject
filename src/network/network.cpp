//
// Created by gs1010 on 29/11/19.
//

#include "network.h"
#include "../preprocessing/preprocessing.h"

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
void Network::Train(const arma::mat &&trainingData, const arma::mat &&trainLabels, int batchSizePercent) {
  //preprocessor.GetTrainingSet(60, 20, 20, std::move(trainingSet));

  int start = 0;
  int end = std::floor((trainingData.n_rows * batchSizePercent) / 100);
  // TODO: split the training set in batch
  for (int i = 1; i <= std::ceil(100 / batchSizePercent); i++) {

    forward(std::move(trainingData.submat(start, 0,
                                          end, trainingData.n_cols - 1)));
    start = end + 1;
    end = i < std::ceil(100 / batchSizePercent) ? std::floor((trainingData.n_rows * batchSizePercent * i) / 100) :
          trainingData.n_rows - 1;
  }
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