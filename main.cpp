#include <iostream>
#include "armadillo"
#include "src/preprocessing/preprocessing.h"
#include "src/network/network.h"

int main() {
  preprocessing a("../../data/monk/monk_dataset.csv", 60, 20, 20);
  Network net;
  net.Add(*new Layer(5, 5));
  net.Add(*new Layer(5, 5));
  net.Add(*new Layer(5, 5));
  net.Init(-1, 1);

  return 0;
}

