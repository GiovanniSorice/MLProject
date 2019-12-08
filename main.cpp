#include <iostream>
#include "armadillo"
#include "src/preprocessing/preprocessing.h"
#include "src/network/network.h"
#include "src/activationFunction/tanhFunction.h"

int main() {
  Preprocessing a("../../data/monk/monk_dataset.csv");
  Network net(a);
  TanhFunction activateFunction;
  Layer firstLayer(15, 15, activateFunction);
  Layer secondLayer(15, 15, activateFunction);
  Layer thirdLayer(15, 15, activateFunction);
  net.Add(firstLayer);
  net.Add(secondLayer);
  net.Add(thirdLayer);
  net.Init(-1, 1);
  net.Train(60, 15);
  return 0;
}