#include <iostream>
#include "armadillo"
#include "src/preprocessing/preprocessing.h"
#include "src/network/network.h"
template<class Matrix>
void print_matrix(Matrix matrix) {
  matrix.print(std::cout);
}
template void print_matrix<arma::mat>(arma::mat matrix);
template void print_matrix<arma::cx_mat>(arma::cx_mat matrix);
int main() {
  Preprocessing a("../../data/monk/monk_dataset.csv", 60, 20, 20);
  std::cout << a.GetTrainingSet().n_elem << std::endl << a.GetValidationSet().n_elem << std::endl
            << a.GetTestSet().n_elem << std::endl;
  std::cout << a.GetTrainingSet() << std::endl << a.GetValidationSet() << std::endl << a.GetTestSet() << std::endl;
  return 0;
}

