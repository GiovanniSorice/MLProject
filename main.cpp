#include <iostream>
#include "armadillo"
#include "src/preprocessing/preprocessing.h"
#include "src/network/network.h"
#include "src/activationFunction/tanhFunction.h"
#include "src/activationFunction/logisticFunction.h"

int main() {

  Preprocessing a("../../data/monk/monk_dataset.csv");
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
  arma::mat
      trainingData = arma::mat(trainingSet.memptr(), trainingSet.n_rows, trainingSet.n_cols - labelCol, false, false);

  Network net;
  TanhFunction tanhFunction;
  LogisticFunction logisticFunction;

  Layer firstLayer(trainingData.n_cols, 15, tanhFunction);
  Layer secondLayer(15, 32, tanhFunction);
  Layer thirdLayer(32, 15, tanhFunction);
  Layer lastLayer(15, 1, logisticFunction);
  net.Add(firstLayer);
  net.Add(secondLayer);
  net.Add(thirdLayer);
  net.Add(lastLayer);

  net.Init(-1, 1);
  net.Train(std::move(trainingData), std::move(trainLabels), 1);

  return 0;
}