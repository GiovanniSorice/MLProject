//
// Created by gs1010 on 21/01/20.
//

#ifndef MLPROJECT_SRC_CROSSVALIDATION_CROSSVALIDATION_H_
#define MLPROJECT_SRC_CROSSVALIDATION_CROSSVALIDATION_H_
#include <armadillo>
#include "../network/network.h"

class CrossValidation {
 public:
  void run(arma::mat dataset,
           arma::mat label,
           int kfold,
           Network net,
           int epoch,
           int batchSize,
           double learningRate,
           double weightDecay,
           double momentum,
           arma::mat meanError
  );

};

#endif //MLPROJECT_SRC_CROSSVALIDATION_CROSSVALIDATION_H_
