//
// Created by gs1010 on 07/12/19.
//

#include "tanhFunction.h"
double TanhFunction::Compute(const double x) {
  return std::tanh(x);
}

void TanhFunction::Compute(const arma::mat &input, arma::mat &&output) {
  output = arma::tanh(input);
}

void TanhFunction::Derive(const arma::mat &&input, arma::mat &&output) {
  output = 1 - arma::pow(arma::tanh(input), 2);
}
double TanhFunction::Derive(const double x) {
  return 1 - std::pow(std::tanh(x), 2);

}
