//
// Created by gs1010 on 29/11/19.
//

#include "network.h"
#include "../preprocessing/preprocessing.h"

Network::Network(LossFunction &lossFunction) : lossFunction(lossFunction) {}

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
                    double learningRate,
                    double momentum) {
  int labelCol = 1;
  //Weighed learning rate
  learningRate = learningRate / ceil(trainingSet.n_rows / batchSize);

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

    train(std::move(trainingData), std::move(trainLabels), batchSize, learningRate, momentum);

    // shuffle the training set for the new epoch
    trainingSet = arma::shuffle(trainingSet);
    // trainingSet.print("Training Set shuffled");
  }
}

//! std::move is used to do a cheap move and not do a deep copy of arma::mat training set
void Network::train(const arma::mat &&trainingData,
                    const arma::mat &&trainLabels,
                    int batchSize,
                    double learningRate,
                    double momentum) {

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
    error(std::move(trainLabels.submat(start, 0,
                                       end, trainLabels.n_cols - 1)),
          std::move(outputActivateBatch),
          std::move(errorBatch));

    backward(std::move(outputActivateBatch), std::move(outputWeightBatch), std::move(errorBatch));

    start = end + 1;
    end = i < std::ceil(trainingData.n_rows / batchSize) ? batchSize * (i + 1) - 1 : trainingData.n_rows - 1;

    updateWeight(learningRate, momentum);
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
    currentLayer.SaveOutputParameter(outputWeight);   // save the activated vectors of the current layer for backpropagation
    currentLayer.Activate(outputWeight, std::move(activateWeight));
    // activateWeight.print("Output activated vector");
  }
  outputActivate = activateWeight;
  // outputActivate.print("Network output");  // print the activation layer output
}

/**
 *  Make the loss function injected object compute the error made by the network for the data passed in
 *
 *  @param outputActivateBatch  Output value produced by the network
 *  @param trainLabelsBatch  Correct value of the data passed in the network
 *  @param errorBatch Error of the current predicted value produced by the network
 * */
void Network::error(const arma::mat &&trainLabelsBatch,
                               arma::mat &&outputActivateBatch,
                               arma::mat &&errorBatch) {
  // outputActivateBatch.print("output activate batch");
  lossFunction.Error(std::move(trainLabelsBatch), std::move(outputActivateBatch), std::move(errorBatch));
  errorBatch.print("errorBatch");
}

/**
 *
 * */
void Network::backward(const arma::mat &&outputActivateBatch,
                       const arma::mat &&outputWeight,
                       const arma::mat &&errorBatch) {
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

void Network::updateWeight(double learningRate, double momentum) {
  for (Layer &currentLayer : net) {
    currentLayer.AdjustWeight(learningRate, momentum);
  }
}

void Network::Test(const arma::mat &&testData, const arma::mat &&testLabels) {
  arma::mat outputActivateBatch;
  arma::mat testDataCopied = testData;

  inference(std::move(testDataCopied),
            std::move(outputActivateBatch));
  //outputActivateBatch.print("outputActivateBatch");

  //testLabels.print("Corrected output");
  //std::cout << testLabels.n_rows << " testLabelsBatch " << testLabels.n_cols << std::endl;

  //arma::mat prova = testLabels - arma::round(outputActivateBatch);

  testLabels.print("testLabels");

  arma::round(outputActivateBatch).print("arma::round(outputActivateBatch)");

}

void Network::inference(arma::mat &&inputData, arma::mat &&outputData) {
  arma::mat activateWeight = arma::mat(inputData.memptr(), inputData.n_rows, inputData.n_cols, false, false);
  for (Layer &currentLayer : net) {
    currentLayer.Forward(std::move(activateWeight), std::move(outputData));

    currentLayer.Activate(outputData, std::move(activateWeight));
  }
  outputData = activateWeight;

}
void Network::TestWithThreshold(const arma::mat &&testData, const arma::mat &&testLabels, double threshold) {
  arma::mat outputActivateBatch;
  arma::mat testDataCopied = testData;

  inference(std::move(testDataCopied),
            std::move(outputActivateBatch));

  outputActivateBatch.print("outputActivateBatch");

  arma::mat thresholdMatrix = arma::ones<arma::mat>(outputActivateBatch.n_rows, outputActivateBatch.n_cols) * threshold;
  arma::mat resultWithThreshold = arma::conv_to<arma::mat>::from(outputActivateBatch > thresholdMatrix);

  (resultWithThreshold - testLabels).print("resultWithThreshold-testLabels");

}