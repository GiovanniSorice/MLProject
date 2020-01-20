#include <iostream>
#include "armadillo"
#include "src/preprocessing/preprocessing.h"
#include "src/network/network.h"
#include "src/activationFunction/tanhFunction.h"
#include "src/activationFunction/logisticFunction.h"
#include "src/lossFunction/meanSquaredError.h"
#include "src/lossFunction/binaryCrossentropy.h"
#include "src/activationFunction/reluFunction.h"
#include "src/load/loadDataset.h"
#include "src/activationFunction/linearFunction.h"

int main() {
  arma::cout.precision(10);
  arma::cout.setf(arma::ios::fixed);
/*
  LoadDataset loadDS;
  loadDS.Load("../../data/monk/monk1_testset.csv");
  loadDS.explodeMonkDataset();
*/

  Preprocessing a("../../data/ML-CUP19-TR_formatted.csv");
  arma::mat trainingSet;
  arma::mat validationSet;
  arma::mat testSet;

  a.GetSplit(100, 0, 0, std::move(trainingSet), std::move(validationSet), std::move(testSet));

  testSet.load("../../data/ML-CUP19-TR_formatted.csv");
  /*
   std::cout << trainingSet.n_rows << " " << trainingSet.n_cols << " " << validationSet.n_rows << " "
            << validationSet.n_cols
            << " " << testSet.n_rows << " " << testSet.n_cols << std::endl;

   */
  int labelCol = 2;
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
  /*
   * std::cout << testData.n_rows << " " << testData.n_cols << " " << validationData.n_rows << " "
            << validationData.n_cols << std::endl;
  */
  MeanSquaredError meanSquaredError;
  BinaryCrossentropy binaryCrossentropy;

  Network net(meanSquaredError);
  //ReluFunction reluFunction;
  TanhFunction tanhFunction;
  LinearFunction linearFunction;
  LogisticFunction logisticFunction;

  // take the first row of the training set for testing purpose
  arma::mat firstRow = trainingSet.row(0);

  Layer firstLayer(trainingSet.n_cols - 1, 4, tanhFunction);
  Layer lastLayer(4, 2, linearFunction);
  net.Add(firstLayer);
  net.Add(lastLayer);

  net.Init(0.7, -0.7);
  net.Train(trainingSet, 1000, 2, 0.1, 0.01, 0.2);

  //net.TestWithThreshold(std::move(trainingData), std::move(trainingLabels), 0.5);
  //net.TestWithThreshold(std::move(validationData), std::move(validationLabels), 0.5);
  net.TestWithThreshold(std::move(testData), std::move(testLabels), 0.5);

  return 0;
}