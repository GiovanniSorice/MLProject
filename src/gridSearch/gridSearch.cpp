//
// Created by gs1010 on 16/01/20.
//

#include "gridSearch.h"
#include "../network/network.h"
#include "../crossValidation/crossValidation.h"
void GridSearch::SetUnitStep(int unit_step) {
  unitStep = unit_step;
}
void GridSearch::SetLearningRateStep(double learning_rate_step) {
  learningRateStep = learning_rate_step;
}
void GridSearch::SetLambdaStep(double lambda_step) {
  lambdaStep = lambda_step;
}
void GridSearch::SetMomentumStep(double momentum_step) {
  momentumStep = momentum_step;
}
void GridSearch::SetLearningRateMin(double learning_rate_min) {
  learningRateMin = learning_rate_min;
}
void GridSearch::SetLearningRateMax(double learning_rate_max) {
  learningRateMax = learning_rate_max;
}
void GridSearch::SetLambdaMin(double lambda_min) {
  lambdaMin = lambda_min;
}
void GridSearch::SetLambdaMax(double lambda_max) {
  lambdaMax = lambda_max;
}
void GridSearch::SetMomentumMin(double momentum_min) {
  momentumMin = momentum_min;
}
void GridSearch::SetMomentumMax(double momentum_max) {
  momentumMax = momentum_max;
}
void GridSearch::SetUnitMin(int unit_min) {
  unitMin = unit_min;
}
void GridSearch::SetUnitMax(int unit_max) {
  unitMax = unit_max;
}
void GridSearch::SetEpochStep(int epoch_step) {
  epochStep = epoch_step;
}
void GridSearch::SetEpochMin(int epoch_min) {
  epochMin = epoch_min;
}
void GridSearch::SetEpochMax(int epoch_max) {
  epochMax = epoch_max;
}

/**
 *   Return the total number of network analyzed by the grid search
 * */
int GridSearch::NetworkAnalyzed() {

  double epochN = (epochMax - epochMin) / epochStep + 1;
  double lambdaN = (lambdaMax - lambdaMin) / lambdaStep + 1;
  double lRN = (learningRateMax - learningRateMin) / learningRateStep + 1;
  double momentumN = (momentumMax - momentumMin) / momentumStep + 1;
  double unitN = (unitMax - unitMin) / unitStep + 1;

  int networkAnalyzed = ceil(epochN * lambdaN * lRN * momentumN * unitN);

  return networkAnalyzed;
}

/***/
void GridSearch::Run(arma::mat dataset, arma::mat label, arma::mat &&result) {
  CrossValidation cross_validation;
  arma::mat error;
  arma::mat trainingDataset = arma::join_rows(dataset, label);
  arma::mat currentError;
  arma::mat testSet;
  testSet.load("../../data/monk/monks1_test_formatted.csv");

  //Split the labels from the test set
  arma::mat testLabels = arma::mat(testSet.memptr() + (testSet.n_cols - 1) * testSet.n_rows,
                                   testSet.n_rows,
                                   1,
                                   false,
                                   false);
  //Split the data from the test test
  arma::mat testData = arma::mat(testSet.memptr(),
                                 testSet.n_rows,
                                 testSet.n_cols - 1,
                                 false,
                                 false);

  int currentRow = 0;
  for (int currentNUnit = unitMin; currentNUnit <= unitMax; currentNUnit += unitStep) {

    Network currNet;
    currNet.SetLossFunction("meanSquaredError");
    Layer firstLayer(dataset.n_cols, currentNUnit, "tanhFunction");
    Layer lastLayer(currentNUnit, label.n_cols, "logisticFunction");
    currNet.Add(firstLayer);
    currNet.Add(lastLayer);
    currNet.Init(1e-4, -1e-4);
    double nDelta;

    for (double currentLambda = lambdaMin; currentLambda <= lambdaMax; currentLambda += lambdaStep) {
      for (double currentMomentum = momentumMin; currentMomentum <= momentumMax; currentMomentum += momentumStep) {
        for (int currentEpoch = epochMin; currentEpoch <= epochMax; currentEpoch += epochStep) {
          for (double currentLearningRate = learningRateMin; currentLearningRate <= learningRateMax;
               currentLearningRate += learningRateStep) {
            error = arma::zeros(1, 1);
            nDelta = 0.0;

            currNet.Clear();
            currNet.Init(1e-4, -1e-4);
            nDelta += currNet.Train(testData,
                                    testLabels,
                                    trainingDataset,
                                    label.n_cols,
                                    currentEpoch,
                                    trainingDataset.n_rows,
                                    currentLearningRate,
                                    currentLambda,
                                    currentMomentum);

            currentError = arma::zeros(1, 1);
            currNet.TestWithThreshold(std::move(testData), std::move(testLabels), std::move(currentError), 0.5);

            /*cross_validation.Run(dataset,
                                 label,
                                 3,
                                 currNet,
                                 currentEpoch,
                                 dataset.n_rows,
                                 currentLearningRate,
                                 currentLambda,
                                 currentMomentum,
                                 std::move(error),
                                 nDelta);
                                 */
            std::cout << " currentNUnit " << currentNUnit << " currentLambda " << currentLambda << " currentMomentum "
                      << currentMomentum
                      << " currentEpoch " << currentEpoch << " currentLearningRate " << currentLearningRate
                      << " error " << currentError.at(0, 0) << " nDelta " << nDelta << std::endl;
/*
            result.at(currentRow, 0) = currentNUnit;
            result.at(currentRow, 1) = currentLambda;
            result.at(currentRow, 2) = currentMomentum;
            result.at(currentRow, 3) = currentEpoch;
            result.at(currentRow, 4) = currentLearningRate;
            result.at(currentRow, 5) = error.at(0, 0);
            result.at(currentRow, 6) = nDelta;
*/
            // increment currentRow for saving, next iteration, the values found
            currentRow++;
          }
        }
      }
    }
  }
  //result.save("grid-search-values.txt", arma::arma_ascii);
}
