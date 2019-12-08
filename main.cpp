#include <iostream>
#include "armadillo"
#include "src/preprocessing/preprocessing.h"
#include "src/network/network.h"

int main() {
  Preprocessing a("../../data/monk/monk_dataset.csv");
  Network net(a);
  a.ReturnSplitDataset(20, 20, 20, arma::mat(), arma::mat(), arma::mat());
  return 0;
}