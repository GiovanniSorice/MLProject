#include <iostream>
#include "armadillo"
#include "src/preprocessing/preprocessing.h"
#include "src/network/network.h"

int main() {
  Preprocessing a("../../data/monk/monk_dataset.csv", 60, 20, 20);
  a.GetTrainingSet().impl_print("Training Set");
  a.GetValidationSet().impl_print("Validation Set");
  a.GetTestSet().impl_print("Test Set");
  return 0;
}