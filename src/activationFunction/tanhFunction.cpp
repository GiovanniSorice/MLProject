//
// Created by gs1010 on 07/12/19.
//

#include "tanhFunction.h"
double tanhFunction::Compute(const double x) {
  return std::tanh(x);
}
void tanhFunction::Derive(const arma::mat &input, arma::mat &output) {
  output = arma::tanh(input);
}
