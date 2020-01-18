//
// Created by gs1010 on 29/11/19.
//

#ifndef MLPROJECT_SRC_NETWORK_H_
#define MLPROJECT_SRC_NETWORK_H_
#include <iostream>
#include "armadillo"
#include "../layer/layer.h"
#include "../preprocessing/preprocessing.h"
#include "../lossFunction/lossFunction.h"

class Network {
 private:
  std::vector<Layer> net;
  arma::mat batch;
  //! Loss function to be optimized in the network
  LossFunction &lossFunction;
  void train(const arma::mat &&trainingData,
             const arma::mat &&trainLabels,
             int batchSize = 32,
             double learningRate = 0.01,
             double weightDecay = 0.0,
             double momentum = 0.0);
  void forward(arma::mat &&batch, arma::mat &&outputActivate, arma::mat &&outputWeight);
  void error(const arma::mat &&trainLabelsBatch,
             arma::mat &&outputActivateBatch,
             arma::mat &&partialDerivativeOutput,
             double weightDecay = 0.0);
  void backward(const arma::mat &&outputActivateBatch, const arma::mat &&outputWeight, const arma::mat &&errorBatch);
  void updateWeight(double learningRate, double weightDecay = 0.0, double momentum = 0.0);
  void inference(arma::mat &&, arma::mat &&);

 public:
  explicit Network(LossFunction &lossFunction);
  void Add(Layer &layer);
  void Init(const double upperBound, const double lowerBound);
  void Train(arma::mat trainingSet,
             int epoch,
             int batchSize = 32,
             double learningRate = 0.01,
             double weightDecay = 0.0,
             double momentum = 0.0);
  void Test(const arma::mat &&testData,
            const arma::mat &&testLabels);
  void TestWithThreshold(const arma::mat &&testData,
                         const arma::mat &&testLabels, double Threshold);
  // TODO: Salvataggio e load (xml? https://www.boost.org/doc/libs/1_71_0/libs/serialization/doc/index.html);

};

#endif //MLPROJECT_SRC_NETWORK_H_

//Togliere preprocessor e passare direttamente le matrici?