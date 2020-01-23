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
#include "src/gridSearch/gridSearch.h"
#include "src/crossValidation/crossValidation.h"

int main() {
  arma::cout.precision(10);
  arma::cout.setf(arma::ios::fixed);
/*
  LoadDataset loadDS;
  loadDS.Load("../../data/monk/monk1_testset.csv");
  loadDS.explodeMonkDataset();
*/

  Preprocessing a("../../data/monk/monks1_train_formatted.csv");
  arma::mat trainingSet;
  arma::mat validationSet;
  arma::mat testSet;

  a.GetSplit(100, 0, 0, std::move(trainingSet), std::move(validationSet), std::move(testSet));

  testSet.load("../../data/monk/monks1_test_formatted.csv");
  /*
   std::cout << trainingSet.n_rows << " " << trainingSet.n_cols << " " << validationSet.n_rows << " "
            << validationSet.n_cols
            << " " << testSet.n_rows << " " << testSet.n_cols << std::endl;

   */
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
/*
  GridSearch gridSearch;
  gridSearch.SetEpochMin(300);
  gridSearch.SetEpochMax(500);
  gridSearch.SetEpochStep(50);
  gridSearch.SetLambdaMin(0.0);
  gridSearch.SetLambdaMax(0.01);
  gridSearch.SetLambdaStep(0.01);
  gridSearch.SetLearningRateMin(0.001);
  gridSearch.SetLearningRateMax(0.07);
  gridSearch.SetLearningRateStep(0.001);
  gridSearch.SetMomentumMin(0);
  gridSearch.SetMomentumMax(0.5);
  gridSearch.SetMomentumStep(0.1);
  gridSearch.SetUnitMin(10);
  gridSearch.SetUnitMax(100);
  gridSearch.SetUnitStep(10);

  gridSearch.run(trainingData, trainingLabels);
*/

  Network net;
  net.SetLossFunction("meanSquaredError");

  Layer firstLayer(trainingSet.n_cols - labelCol, 4, "tanhFunction");
  Layer lastLayer(4, 1, "logisticFunction");
  net.Add(firstLayer);
  net.Add(lastLayer);

  net.Init(1e-3, -1e-3);
  /*
  net.Train(trainingSet, trainingLabels.n_cols, 800, 128, 0.9, 0, 0.5);
  net.TestWithThreshold(std::move(testData), std::move(testLabels), 0.5);
*/

  return 0;
}