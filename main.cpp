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
  double learningRateMin = 0.1;
  double learningRateMax = 1;
  double learningRateStep = 0.1;
  double lambdaMin = 0;
  double lambdaMax = 0.001;
  double lambdaStep = 0.0002;
  double momentumMin = 0;
  double momentumMax = 0.5;
  double momentumStep = 0.1;
  int unitMin = 3;
  int unitMax = 10;
  int unitStep = 1;
  int epochMin = 800;
  int epochMax = 800;
  int epochStep = 1;

  GridSearch gridSearch;
  gridSearch.SetEpochMin(epochMin);
  gridSearch.SetEpochMax(epochMax);
  gridSearch.SetEpochStep(epochStep);
  gridSearch.SetLambdaMin(lambdaMin);
  gridSearch.SetLambdaMax(lambdaMax);
  gridSearch.SetLambdaStep(lambdaStep);
  gridSearch.SetLearningRateMin(learningRateMin);
  gridSearch.SetLearningRateMax(learningRateMax);
  gridSearch.SetLearningRateStep(learningRateStep);
  gridSearch.SetMomentumMin(momentumMin);
  gridSearch.SetMomentumMax(momentumMax);
  gridSearch.SetMomentumStep(momentumStep);
  gridSearch.SetUnitMin(unitMin);
  gridSearch.SetUnitMax(unitMax);
  gridSearch.SetUnitStep(unitStep);

  int netAnalyzed = gridSearch.NetworkAnalyzed();
  arma::mat result = arma::zeros(netAnalyzed, 5);   // 4 hyperparams and error
  gridSearch.Run(trainingData, trainingLabels, std::move(result));

/*
  Network net;
  net.SetLossFunction("meanSquaredError");

  Layer firstLayer(trainingSet.n_cols - labelCol, 4, "tanhFunction");
  Layer lastLayer(4, 1, "logisticFunction");
  net.Add(firstLayer);
  net.Add(lastLayer);

  net.Init(1e-3, -1e-3);

  net.Train(trainingSet, trainingLabels.n_cols, 800, 128, 0.9, 0, 0.5);
  net.TestWithThreshold(std::move(testData), std::move(testLabels), 0.5);

  CrossValidation cross_validation;
  arma::mat error = arma::zeros(1, trainingLabels.n_cols);
  cross_validation.run(trainingData,
                       trainingLabels,
                       3,
                       net,
                       800,
                       trainingData.n_rows,
                       0.9,
                       0,
                       0.5,
                       std::move(error));
*/
  return 0;
}