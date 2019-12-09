#include <iostream>
#include "armadillo"
#include "src/preprocessing/preprocessing.h"
#include "src/network/network.h"
#include "src/activationFunction/tanhFunction.h"

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
  TanhFunction activateFunction;
  Layer firstLayer(trainingData.n_rows, 15, activateFunction);
  Layer secondLayer(15, 15, activateFunction);
  Layer thirdLayer(15, 15, activateFunction);
  net.Add(firstLayer);
  net.Add(secondLayer);
  net.Add(thirdLayer);
  net.Init(-1, 1);
  net.Train(std::move(trainingData), std::move(trainLabels));

  return 0;
}