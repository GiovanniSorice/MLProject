//
// Created by gs1010 on 29/11/19.
//

#include "network.h"
Network::Network() = default;
void Network::Add(Layer &layer) {
  net.push_back(layer);
}
void Network::Init(const double upperBound = 1, const double lowerBound = -1) {
  for (auto &i : net) {
    i.Init(upperBound, lowerBound);
  }
}
void Network::Forward() {
  /*
  arma::mat = trainingSet;
  for (auto &i : net) {
    i.Forward()
  }*/
}
