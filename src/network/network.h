//
// Created by gs1010 on 29/11/19.
//

#ifndef MLPROJECT_SRC_NETWORK_H_
#define MLPROJECT_SRC_NETWORK_H_
#include <iostream>
#include "armadillo"
#include "../layer/layer.h"
#include "../preprocessing/preprocessing.h"

class Network {
 public:
  explicit Network() = default;
 private:
  std::vector<Layer> net;
  arma::mat batch;
  void forward(arma::mat &&batch);
 public:
  void Add(Layer &layer);
  void Init(const double upperBound, const double lowerBound);
  void Train(const arma::mat &&trainingData, const arma::mat &&trainLabels, int batchSizePercent = 15);
  // TODO: Metodo Fit o Train in network a cui passo le epoche da fare
  // TODO: Salvataggio e load (xml? https://www.boost.org/doc/libs/1_71_0/libs/serialization/doc/index.html);

};

#endif //MLPROJECT_SRC_NETWORK_H_

//Togliere preprocessor e passare direttamente le matrici?