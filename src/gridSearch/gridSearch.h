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
  int NetworkAnalyzed();
  void Run(arma::mat dataset,
           arma::mat label, arma::mat &&result);
  void SetLearningRate(double learning_rate_min, double learning_rate_max, double learning_rate_step);
  void SetLambda(double lambda_min, double lambda_max, double lambda_step);
  void SetMomentum(double momentum_min, double momentum_max, double momentum_step);
  void SetUnit(int unit_min, int unit_max, int unit_step);
  void SetEpoch(int epoch_min, int epoch_max, int epoch_step);

};

#endif //MLPROJECT_SRC_GRIDSEARCH_GRIDSEARCH_H_
