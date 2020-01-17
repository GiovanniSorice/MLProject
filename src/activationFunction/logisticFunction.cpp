//
// Created by gs1010 on 09/12/19.
//

#include "logisticFunction.h"
double LogisticFunction::Compute(const double x) {
  if (x < arma::datum::log_max) {
    if (x > -arma::datum::log_max)
      return 1.0 / (1.0 + std::exp(-x));

    return 0.0;
  }

  return 1.0;
}
void LogisticFunction::Compute(const arma::mat &input, arma::mat &&output) {
  output = 1.0 / (1.0 + arma::exp(-input));
  //output.print("Logistic output");
}
void LogisticFunction::Derive(const arma::mat &&input, arma::mat &&output) {
  //input.print("input");
  arma::mat activatedInput;
  Compute(std::move(input), std::move(activatedInput));
  output = activatedInput % (1.0 - activatedInput);
  //output.print("derived output");
}
double LogisticFunction::Derive(const double x) {
  return Compute(x) * (1.0 - Compute(x));
}
