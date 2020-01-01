//
// Created by checco on 01/01/20.
//

#include "reluFunction.h"
double ReluFunction::Compute(const double x) {
  return std::max(0.0, x);
}

void ReluFunction::Compute(const arma::mat &input, arma::mat &&output) {
  output.zeros(input.n_rows, input.n_cols);
  output = arma::max(output, input);
}

void ReluFunction::Derive(const arma::mat &&input, arma::mat &&output) {
  output.set_size(arma::size(input));

  for (size_t i = 0; i < input.n_elem; i++) {
    output(i) = Derive(input(i));
  }
}

/**
 *  Derivative is 0 if x < 0, 1 if x > 0, and undefined if x = 0
 * */
double ReluFunction::Derive(const double x) {
  return (double) (x > 0);
}
