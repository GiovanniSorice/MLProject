//
// Created by gs1010 on 16/01/20.
//

#include "gridSearch.h"
#include "../network/network.h"
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

void GridSearch::run(arma::mat dataset, arma::mat label) {
  arma::mat joinedDataset = arma::join_rows(dataset, label);

  for (int currentNUnit = unitMin; currentNUnit < unitMax; currentNUnit += unitStep) {
    Network currNet;
    currNet.SetLossFunction("meanSquaredError");
    Layer firstLayer(dataset.n_cols - label.n_cols, currentNUnit, "tanhFunction");
    Layer lastLayer(currentNUnit, label.n_cols, "linearFunction");
    currNet.Add(firstLayer);
    currNet.Add(lastLayer);
    currNet.Init(0.7, -0.7);

    for (double currentLambda = lambdaMin; currentLambda <= lambdaMax; currentLambda += lambdaStep) {
      for (double currentMomentum = momentumMin; currentMomentum <= momentumMax; currentMomentum += momentumStep) {
        for (int currentEpoch = epochMin; currentMomentum <= epochMax; currentMomentum += epochStep) {
          for (double currentLearningRate = learningRateMin; currentLearningRate <= learningRateMax;
               currentLearningRate += learningRateStep) {
            currNet.Train(joinedDataset,
                          currentEpoch,
                          label.n_cols,
                          1,
                          currentLearningRate,
                          currentLambda,
                          currentMomentum);
            currNet.Test(std::move(dataset), std::move(label));
          }
        }
      }
    }
  }

}
