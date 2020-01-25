//
// Created by checco on 25/01/20.
//

#ifndef MLPROJECT_SRC_GRIDSEARCH_PARALLELGRIDSEARCH_H_
#define MLPROJECT_SRC_GRIDSEARCH_PARALLELGRIDSEARCH_H_
#include <armadillo>
#include "gridSearch.h"
#include <thread>
class ParallelGridSearch {
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
  int totalThreadNumber;
  std::vector<std::thread> gridSearchThreads;
  std::vector<arma::mat> resultMatrix;
  std::vector<GridSearch> gridSearches;
  void setNumberThread(int totalThread);
  double learningRateInterval();
  double momentumInterval();
  double lambdaInterval();
  int epochInterval();
  int unitInterval();
  void setGridsSearch(int totalNetworkAnalyzed);
  void saveResult();
 public:
  void Run(arma::mat dataset, arma::mat label);
  int NetworkAnalyzed();
  void SetUnitStep(int unit_step);
  void SetLearningRateStep(double learning_rate_step);
  void SetLambdaStep(double lambda_step);
  void SetMomentumStep(double momentum_step);
  void SetEpochStep(int epoch_step);
  void SetLearningRateMin(double learning_rate_min);
  void SetLearningRateMax(double learning_rate_max);
  void SetLambdaMin(double lambda_min);
  void SetLambdaMax(double lambda_max);
  void SetMomentumMin(double momentum_min);
  void SetMomentumMax(double momentum_max);
  void SetUnitMin(int unit_min);
  void SetUnitMax(int unit_max);
  void SetEpochMin(int epoch_min);
  void SetEpochMax(int epoch_max);
};

#endif //MLPROJECT_SRC_GRIDSEARCH_PARALLELGRIDSEARCH_H_
