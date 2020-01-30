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
  //! Data preprocessing.
  Preprocessing cupPreprocessing("../../data/ML-CUP19-TR_formatted.csv");
  arma::mat trainingSet;
  arma::mat validationSet;
  arma::mat testSet;

  cupPreprocessing.GetSplit(60, 20, 20, std::move(trainingSet), std::move(validationSet), std::move(testSet));
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

  //! Grid search implementation (the parallel one can be also used
  //! changing GridSearch class with ParallelGridSearch class)
  /*
  double learningRateMin = 0.0001;
  double learningRateMax = 0.001;
  double learningRateStep = 0.00005;
  double lambdaMin = 0;
  double lambdaMax = 0.001;
  double lambdaStep = 0.001;
  double momentumMin = 0.8;
  double momentumMax = 0.8;
  double momentumStep = 0.2;
  int unitMin = 100;
  int unitMax = 150;
  int unitStep = 50;
  int epochMin = 8000;
  int epochMax = 8000;
  int epochStep = 1;

  GridSearch gridSearch;
  gridSearch.SetLambda(lambdaMin, lambdaMax, lambdaStep);
  gridSearch.SetLearningRate(learningRateMin, learningRateMax, learningRateStep);
  gridSearch.SetMomentum(momentumMin, momentumMax, momentumStep);
  gridSearch.SetUnit(unitMin, unitMax, unitStep);
  gridSearch.SetEpoch(epochMin, epochMax, epochStep);
  arma::mat result = arma::zeros(gridSearch.NetworkAnalyzed(), 8);
  gridSearch.Run(trainingData, trainingLabels, std::move(result));
  */



  //! ML CUP network, training and testing
  Network cupNetwork;
  cupNetwork.SetLossFunction("meanSquaredError");
  Layer firstLayer(trainingSet.n_cols - labelCol, 75, "tanhFunction");
  Layer lastLayer(75, 2, "linearFunction");
  cupNetwork.Add(firstLayer);
  cupNetwork.Add(lastLayer);
  cupNetwork.Init(0.7, -0.7);
  cupNetwork.Train(validationData,
                   validationLabels,
                   trainingSet,
                   trainingLabels.n_cols,
                   15000,
                   trainingSet.n_rows,
                   0.005,
                   0.00001,
                   0.6);

  arma::mat MEE;
  cupNetwork.Test(std::move(validationData), std::move(validationLabels), std::move(MEE));
  MEE.print("errore finale");

  //! Cross validation implementation
  /*
  CrossValidation cross_validation;
  arma::mat error = arma::zeros(1, trainingLabels.n_cols);
  double nDelta = 0;
  cross_validation.Run(trainingData,
                       trainingLabels,
                       3,
                       cupNetwork,
                       15000,
                       trainingData.n_rows,
                       0.005,
                       0.0001,
                       0.8,
                       std::move(error),
                       nDelta);
  */


  return 0;
}
