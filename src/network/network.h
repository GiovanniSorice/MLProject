//
// Created by gs1010 on 29/11/19.
//

#ifndef MLPROJECT_SRC_NETWORK_H_
#define MLPROJECT_SRC_NETWORK_H_
#include <iostream>
#include "armadillo"
#include "../layer/layer.h"

class Network {
 public:
  Network();
 private:
  std::vector<Layer> net;
  arma::mat batch;
 public:
  void Add(Layer &layer);
  void Init(const double upperBound, const double lowerBound);
  void Forward();
// Metodo Fit o Train in network a cui passo le epoche da fare
// Inizializzazione della rete con random weight
// Salvataggio e load (xml? https://www.boost.org/doc/libs/1_71_0/libs/serialization/doc/index.html);
// Classe dataset che gestisce la lettura e la suddivisione del dataset;
// Add di un layer
// Togliere un layer
// Funzione di test
//
};

#endif //MLPROJECT_SRC_NETWORK_H_
