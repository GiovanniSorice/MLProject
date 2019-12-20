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

/**
 * Split the training set in training data and training labels and feed them in the network.
 *
 *
 * @param trainingSet Deep copy of the training set
 * @param epoch Number of total shuffling of the training set feed in the network
 * @param batchSize Number of the example feed in the network for each forward pass
 * @param learningRate Adjust weight parameter
 * */
void Network::Train(arma::mat trainingSet,
                    int epoch,
                    int batchSize,
                    double learningRate) {
  int labelCol = 1;
  for (int currentEpoch = 1; currentEpoch <= epoch; currentEpoch++) {

    // Split the data from the training set.
    arma::mat trainLabels = arma::mat(trainingSet.memptr() + (trainingSet.n_cols - labelCol) * trainingSet.n_rows,
                                      trainingSet.n_rows,
                                      labelCol,
                                      false,
                                      false);

    // Split the labels from the training set.
    arma::mat trainingData = arma::mat(trainingSet.memptr(),
                                       trainingSet.n_rows,
                                       trainingSet.n_cols - labelCol,
                                       false,
                                       false);

    train(std::move(trainingData), std::move(trainLabels), batchSize, learningRate);

    // shuffle the training set for the new epoch
    trainingSet = arma::shuffle(trainingSet);
    // trainingSet.print("Training Set shuffled");
  }
}

//! std::move is used to do a cheap move and not do a deep copy of arma::mat training set
void Network::train(const arma::mat &&trainingData,
                    const arma::mat &&trainLabels,
                    int batchSize,
                    double learningRate) {

  //TODO: Da verificare se il learningRate non debba essere adattato per il numero di batch
  int start = 0;
  int end = batchSize - 1;
  arma::mat outputWeightBatch;
  arma::mat outputActivateBatch;
  arma::mat errorBatch;

  for (int i = 1; i <= std::ceil(trainingData.n_rows / batchSize); i++) {

      forward(std::move(trainingData.submat(start, 0,
                                            end, trainingData.n_cols - 1)),
              std::move(outputActivateBatch),
              std::move(outputWeightBatch));

      meanSquaredError(std::move(trainLabels.submat(start, 0,
                                                    end, trainLabels.n_cols - 1)),
                       std::move(outputActivateBatch),
                       std::move(errorBatch));

    backward(std::move(outputActivateBatch), std::move(outputWeightBatch), std::move(errorBatch));

      start = end + 1;
    end = i < std::ceil(trainingData.n_rows / batchSize) ? batchSize * (i + 1) - 1 : trainingData.n_rows - 1;

    updateWeight(learningRate);
  }
}

/**
 *  Iterate over the raw in the batch and pass them to the network
 *  TODO: outputWeight può essere tolto perchè è già salvato nel layer tramite currentLayer.SaveOutputParameter(outputWeight)
 * */
void Network::forward(arma::mat &&batch, arma::mat &&outputActivate, arma::mat &&outputWeight) {
  arma::mat activateWeight = arma::mat(batch.memptr(), batch.n_rows, batch.n_cols, false, false);
  for (Layer &currentLayer : net) {
    // activateWeight.print("Input feed vector");
    currentLayer.SaveInputParameter(activateWeight);    // save the input vector of the layer
    currentLayer.Forward(std::move(activateWeight), std::move(outputWeight));
    // outputWeight.print("Weight plus bias vector");
    currentLayer.Activate(outputWeight, std::move(activateWeight));
    currentLayer.SaveOutputParameter(activateWeight);   // save the activated vectors of the current layer for backpropagation
    // activateWeight.print("Output activated vector");
  }
  outputActivate = activateWeight;
  // outputActivate.print("Network output");  // print the activation layer output

}

void Network::meanSquaredError(const arma::mat &&trainLabelsBatch,
                               arma::mat &&outputActivateBatch,
                               arma::mat &&errorBatch) {
  errorBatch = arma::mean(arma::pow(trainLabelsBatch - outputActivateBatch, 2));
  errorBatch.print("Network output error");
}
void Network::backward(const arma::mat &&outputActivateBatch,
                       const arma::mat &&outputWeight,
                       const arma::mat &&errorBatch) {
  arma::mat gradient;
  auto currentLayer = net.rbegin();
  currentLayer->OutputLayerGradient(std::move(errorBatch));
  arma::mat currentGradientWeight;
  currentLayer->GetSummationWeight(std::move(currentGradientWeight));

  currentLayer++;
  // Iterate from the precedent Layer of the tail to the head
  for (; currentLayer != net.rend(); currentLayer++) {
    currentLayer->Gradient(std::move(currentGradientWeight));
    currentLayer->GetSummationWeight(std::move(currentGradientWeight));
  }
}

void Network::updateWeight(double learningRate) {
  for (Layer &currentLayer : net) {
    currentLayer.AdjustWeight(learningRate);
  }
}

void Network::Test(const arma::mat &&testData, const arma::mat &&testLabels, int batchSize) {
  arma::mat outputWeightBatch;
  arma::mat outputActivateBatch;

  int start = 0;
  int end = batchSize - 1;

  for (int i = 1; i <= std::ceil(testData.n_rows / batchSize); i++) {
    forward(std::move(testData.submat(start, 0,
                                      end, testData.n_cols - 1)),
            std::move(outputActivateBatch),
            std::move(outputWeightBatch));

    (arma::mean(outputActivateBatch)).print("Network predicted result");
    arma::mat testLabelsBatch = testLabels.submat(start, 0, end, testLabels.n_cols - 1);
    testLabelsBatch.print("Corrected output");

    start = end + 1;
    end = i < std::ceil(testData.n_rows / batchSize) ? batchSize * (i + 1) - 1 : testData.n_rows - 1;
  }
}