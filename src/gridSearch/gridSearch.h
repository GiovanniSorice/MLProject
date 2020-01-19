//
// Created by gs1010 on 16/01/20.
//

#ifndef MLPROJECT_SRC_GRIDSEARCH_GRIDSEARCH_H_
#define MLPROJECT_SRC_GRIDSEARCH_GRIDSEARCH_H_
#include <armadillo>

class gridSearch {
 private:
  int unitStep;
  double learningRateStep;
  double lamdaStep;
  double momentumStep;
 public:
  gridSearch(int unitStep, double learningRateStep, double lamdaStep, double momentumStep);
  void run(arma::mat dataset, arma::mat label, int unit, double learningRate, double lamda, double momentum);
};

#endif //MLPROJECT_SRC_GRIDSEARCH_GRIDSEARCH_H_
