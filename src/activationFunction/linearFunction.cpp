//
// Created by gs1010 on 19/01/20.
//

#include "linearFunction.h"
double LinearFunction::Compute(const double x) {
  return x;
}
void LinearFunction::Compute(const arma::mat &input, arma::mat &&output) {
  output = input;
}
void LinearFunction::Derive(const arma::mat &&input, arma::mat &&output) {
  output = arma::ones(input.n_rows, input.n_cols);
}
double LinearFunction::Derive(const double x) {
  return 1;
}
