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
  std::cout << trainingSet.n_rows << " " << trainingSet.n_cols << " " << validationSet.n_rows << " "
            << validationSet.n_cols
            << " " << testSet.n_rows << " " << testSet.n_cols << std::endl;
  int labelCol = 1;


  // Split the data from the training set.
  arma::mat trainingLabels = arma::mat(trainingSet.memptr() + (trainingSet.n_cols - labelCol) * trainingSet.n_rows,
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
  arma::mat
      validationLabels = arma::mat(validationSet.memptr() + (validationSet.n_cols - labelCol) * validationSet.n_rows,
                                   validationSet.n_rows,
                                   labelCol,
                                   false,
                                   false);

  //Split the data from the test test
  arma::mat validationData = arma::mat(validationSet.memptr(),
                                       validationSet.n_rows,
                                       validationSet.n_cols - labelCol,
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
  std::cout << testData.n_rows << " " << testData.n_cols << " " << validationData.n_rows << " "
            << validationData.n_cols << std::endl;

  Network net;
  TanhFunction tanhFunction;
  LogisticFunction logisticFunction;

  Layer firstLayer(trainingSet.n_cols - 1, 3, tanhFunction);
  Layer lastLayer(3, 1, logisticFunction);
  net.Add(firstLayer);
  net.Add(lastLayer);

  net.Init(-1e-4, 1e-4);
  net.TrainWM(trainingSet, 290, 32, 0.9);
  net.TestWithThreshold(std::move(trainingData), std::move(trainingLabels), 0.5);

  net.TestWithThreshold(std::move(validationData), std::move(validationLabels), 0.5);

  net.TestWithThreshold(std::move(testData), std::move(testLabels), 0.5);

  return 0;
}