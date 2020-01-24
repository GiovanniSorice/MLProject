//
// Created by gs1010 on 16/01/20.
//

#ifndef MLPROJECT_SRC_GRIDSEARCH_GRIDSEARCH_H_
#define MLPROJECT_SRC_GRIDSEARCH_GRIDSEARCH_H_
#include <armadillo>

class GridSearch {
 private:
  int unitStep;
  double learningRateStep;
  double lambdaStep;
  double momentumStep;
  int epochStep;
  double learningRateMin;
  double learningRateMax;
  double lambdaMin;
  double lambdaMax;
  double momentumMin;
  double momentumMax;
  int unitMin;
  int unitMax;
  int epochMin;
  int epochMax;

 public:
  void SetEpochStep(int epoch_step);
  void SetEpochMin(int epoch_min);
  void SetEpochMax(int epoch_max);
  void run(arma::mat dataset,
           arma::mat label, arma::mat
           &&result);
  void SetUnitStep(int unit_step);
  void SetLearningRateStep(double learning_rate_step);
  void SetLambdaStep(double lambda_step);
  void SetMomentumStep(double momentum_step);
  void SetLearningRateMin(double learning_rate_min);
  void SetLearningRateMax(double learning_rate_max);
  void SetLambdaMin(double lambda_min);
  void SetLambdaMax(double lambda_max);
  void SetMomentumMin(double momentum_min);
  void SetMomentumMax(double momentum_max);
  void SetUnitMin(int unit_min);
  void SetUnitMax(int unit_max);
};

#endif //MLPROJECT_SRC_GRIDSEARCH_GRIDSEARCH_H_
