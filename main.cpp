#include <iostream>
#include "armadillo"
#include "src/preprocessing/preprocessing.h"
#include "src/network/network.h"
#include "src/activationFunction/tanhFunction.h"
#include "src/activationFunction/logisticFunction.h"
#include "src/load/loadDataset.h"

int main() {
  Preprocessing a("../../data/monk/monks-formatted.csv");
  arma::mat trainingSet;
  arma::mat validationSet;
  arma::mat testSet;

  a.GetSplit(60, 20, 20, std::move(trainingSet), std::move(validationSet), std::move(testSet));
  int labelCol = 1;

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

  //Split the labels from the test set
  arma::mat testLabels = arma::mat(testSet.memptr() + (testSet.n_cols - labelCol) * testSet.n_rows,
                                   testSet.n_rows,
                                   labelCol,
                                   false,
                                   false);

  //Split the data from the test test
  arma::mat testData = arma::mat(testSet.memptr(),
                                 testSet.n_rows,
                                 testSet.n_cols - labelCol,
                                 false,
                                 false);

  Network net;
  TanhFunction tanhFunction;
  LogisticFunction logisticFunction;

  Layer firstLayer(trainingData.n_cols, 3, tanhFunction);
  Layer secondLayer(3, 4, tanhFunction);
  Layer lastLayer(4, 1, logisticFunction);
  net.Add(firstLayer);
  net.Add(secondLayer);
  net.Add(lastLayer);

  net.Init(-1, 1);
  net.Train(std::move(trainingData), std::move(trainLabels), 1, 1);
  net.Test(std::move(testData), std::move(testLabels));
  return 0;
}