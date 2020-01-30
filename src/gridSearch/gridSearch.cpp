//
// Created by gs1010 on 16/01/20.
//

#include "gridSearch.h"
#include "../network/network.h"
#include "../crossValidation/crossValidation.h"

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
  int currentRow = 0;
  for (int currentNUnit = unitMin; currentNUnit <= unitMax; currentNUnit += unitStep) {

    Network currNet;
    currNet.SetLossFunction("meanSquaredError");
    Layer firstLayer(dataset.n_cols, currentNUnit, "tanhFunction");
    Layer secondLayer(currentNUnit, currentNUnit / 2, "tanhFunction");
    Layer thirdLayer(currentNUnit / 2, currentNUnit / 3, "tanhFunction");
    Layer lastLayer(currentNUnit / 3, label.n_cols, "linearFunction");
    currNet.Add(firstLayer);
    currNet.Add(secondLayer);
    currNet.Add(thirdLayer);
    currNet.Add(lastLayer);
    currNet.Init(0.7, -0.7);
    double nDelta;

    for (double currentLambda = lambdaMin; currentLambda <= lambdaMax; currentLambda += lambdaStep) {
      for (double currentMomentum = momentumMin; currentMomentum <= momentumMax; currentMomentum += momentumStep) {
        for (int currentEpoch = epochMin; currentEpoch <= epochMax; currentEpoch += epochStep) {
          for (double currentLearningRate = learningRateMin; currentLearningRate <= learningRateMax;
               currentLearningRate += learningRateStep) {
            error = arma::zeros(1, 1);
            nDelta = 0.0;
            cross_validation.Run(dataset,
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

            result.at(currentRow, 0) = currentNUnit;
            result.at(currentRow, 1) = currentLambda;
            result.at(currentRow, 2) = currentMomentum;
            result.at(currentRow, 3) = currentEpoch;
            result.at(currentRow, 4) = currentLearningRate;
            result.at(currentRow, 5) = error.at(0, 0);
            result.at(currentRow, 6) = nDelta;

            // increment currentRow for saving, next iteration, the values found
            currentRow++;
          }
        }
      }
    }
  }
  result.save("grid-search-values" + std::to_string(std::rand()) + ".txt", arma::arma_ascii);
}

void GridSearch::SetLearningRate(double learning_rate_min, double learning_rate_max, double learning_rate_step) {
  learningRateMin = learning_rate_min;
  learningRateMax = learning_rate_max;
  learningRateStep = learning_rate_step;
}
void GridSearch::SetLambda(double lambda_min, double lambda_max, double lambda_step) {
  lambdaMin = lambda_min;
  lambdaMax = lambda_max;
  lambdaStep = lambda_step;
}
void GridSearch::SetUnit(int unit_min, int unit_max, int unit_step) {
  unitMax = unit_max;
  unitMin = unit_min;
  unitStep = unit_step;
}
void GridSearch::SetMomentum(double momentum_min, double momentum_max, double momentum_step) {
  momentumMin = momentum_min;
  momentumMax = momentum_max;
  momentumStep = momentum_step;
}
void GridSearch::SetEpoch(int epoch_min, int epoch_max, int epoch_step) {
  epochMin = epoch_min;
  epochMax = epoch_max;
  epochStep = epoch_step;
}
