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
void Network::Train(const arma::mat &&trainingData,
                    const arma::mat &&trainLabels,
                    int batchSize,
                    double learningRate) {
  //TODO: Da verificare se il learningRate non debba essere adatatto per il numero di batch
  int start = 0;
  int end = batchSize - 1;
  arma::mat outputWeightBatch;
  arma::mat outputActivateBatch;
  arma::mat errorBatch;

  for (int i = 1; i <= std::ceil(trainingData.n_rows / batchSize); i++) {

    arma::mat a = trainingData.submat(start, 0,
                                      end, trainingData.n_cols - 1);

    forward(std::move(trainingData.submat(start, 0,
                                          end, trainingData.n_cols - 1)),
            std::move(outputActivateBatch),
            std::move(outputWeightBatch));

    meanSquaredError(std::move(trainLabels.submat(start, 0,
                                                  end, trainLabels.n_cols - 1)),
                     std::move(outputActivateBatch),
                     std::move(errorBatch));

    backward(std::move(outputActivateBatch), std::move(outputWeightBatch), std::move(errorBatch), learningRate);

    start = end + 1;
    end =
        i < std::ceil(trainingData.n_rows / batchSize) ? batchSize * (i + 1) - 1 :
        trainingData.n_rows - 1;
  }
}

/**
 *  Iterate over the raw in the batch and pass them to the network
 *  TODO: outputWeight può essere tolto perchè è già salvato nel layer tramite currentLayer.SaveOutputParameter(outputWeight)
 * */
void Network::forward(arma::mat &&batch, arma::mat &&outputActivate, arma::mat &&outputWeight) {
  arma::mat activateWeight = arma::mat(batch.memptr(), batch.n_rows, batch.n_cols, false, false);
  for (Layer &currentLayer : net) {
    currentLayer.Forward(std::move(activateWeight), std::move(outputWeight));
    currentLayer.SaveOutputParameter(outputWeight);
    currentLayer.Activate(outputWeight, std::move(activateWeight));
  }
  outputActivate = std::move(activateWeight);
}
void Network::meanSquaredError(const arma::mat &&trainLabelsBatch,
                               arma::mat &&outputActivateBatch,
                               arma::mat &&errorBatch) {
  errorBatch = arma::mean(arma::pow(trainLabelsBatch - outputActivateBatch, 2));
}
void Network::backward(const arma::mat &&outputActivateBatch,
                       const arma::mat &&outputWeight,
                       const arma::mat &&errorBatch,
                       double learningRate) {
  arma::mat gradient;
  std::vector<Layer>::reverse_iterator currentLayer = net.rbegin();
  currentLayer->OutputLayerGradient(std::move(outputWeight), std::move(errorBatch), std::move(gradient));
  arma::mat summationGradientWeight = gradient * currentLayer->GetWeight();
  currentLayer++;
  // Iterate from the precedent Layer of the tail to the head
  for (; currentLayer != net.rend(); currentLayer++) {
    currentLayer->Gradient(std::move(summationGradientWeight), std::move(gradient));
    summationGradientWeight += gradient;
    summationGradientWeight.print("summationGradientWeight");
  }
}
