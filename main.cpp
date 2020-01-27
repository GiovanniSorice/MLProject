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
#include "src/gridSearch/parallelGridSearch.h"

int main() {
  arma::cout.precision(10);
  arma::cout.setf(arma::ios::fixed);

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
  double learningRateMin = 0.1;
  double learningRateMax = 0.9;
  double learningRateStep = 0.1;
  double lambdaMin = 0;
  double lambdaMax = 0.001;
  double lambdaStep = 0.0001;
  double momentumMin = 0.0;
  double momentumMax = 0.9;
  double momentumStep = 0.1;
  int unitMin = 3;
  int unitMax = 5;
  int unitStep = 1;
  int epochMin = 800;
  int epochMax = 800;
  int epochStep = 1;

  ParallelGridSearch gridSearch;
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
  std::cout << "netAnalyzed" << netAnalyzed << std::endl;
  //arma::mat result = arma::zeros(netAnalyzed, 7);   // 4 hyperparams and error
  arma::mat result = arma::zeros(netAnalyzed, 7);
  gridSearch.Run(trainingData, trainingLabels);

*/



  Network net;
  net.SetLossFunction("meanSquaredError");
  Layer firstLayer(trainingSet.n_cols - labelCol, 3, "tanhFunction");
  Layer lastLayer(3, 1, "logisticFunction");
  net.Add(firstLayer);
  net.Add(lastLayer);

  net.Init(1e-3, -1e-3);

  net.Train(testData,
            testLabels,
            trainingSet,
            trainingLabels.n_cols,
            800,
            trainingSet.n_rows,
            0.9,
            0,
            0.7);
  arma::mat mat = arma::zeros(1, 1);
  net.TestWithThreshold(std::move(testData), std::move(testLabels), std::move(mat), 0.5);
  //net.Test(std::move(testData), std::move(testLabels), std::move(mat));
 mat.print("errore finale");

/*
  CrossValidation cross_validation;
  arma::mat error = arma::zeros(1, trainingLabels.n_cols);
  double nDelta = 0;
  cross_validation.Run(trainingData,
                       trainingLabels,
                       3,
                       net,
                       15000,
                       trainingData.n_rows,
                       0.000975,
                       0,
                       0.8,
                       std::move(error),
                       nDelta);

  net.Test(std::move(validationData), std::move(validationLabels), std::move(mat));
  mat.print("error");
  */
  return 0;
}