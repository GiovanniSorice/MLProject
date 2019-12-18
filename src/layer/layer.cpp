//
// Created by checco on 26/11/19.
//

#include "layer.h"
const arma::mat &Layer::GetWeight() const {
  return weight;
}
const arma::mat &Layer::GetBias() const {
  return bias;
}
const arma::mat &Layer::GetDelta() const {
  return delta;
}
const arma::mat &Layer::GetGradient() const {
  return gradient;
}
const arma::mat &Layer::GetInputParameter() const {
  return inputParameter;
}
const arma::mat &Layer::GetOutputParameter() const {
  return outputParameter;
}

Layer::Layer(const int inSize, const int outSize, ActivationFunction &activationFunction)
    : inSize(inSize), outSize(outSize), activationFunction(activationFunction) {
}

// TODO: Da testare
void Layer::Forward(const arma::mat &&input, arma::mat &&output) {
  output = input * weight;
  output.each_row() += bias;
}
void Layer::Backward(const arma::mat &&input, arma::mat &&gy, arma::mat &&g) {

}
// TODO: Da testare
void Layer::OutputLayerGradient(const arma::mat &&error) {
  arma::mat firstDerivativeActivation;
  activationFunction.Derive(std::move(outputParameter), std::move(firstDerivativeActivation));
  firstDerivativeActivation = arma::mean(firstDerivativeActivation);
  gradient = arma::mean(error % firstDerivativeActivation);
  gradient.print("gradient");
}
void Layer::Initialize() {
  weight = arma::mat(inSize, outSize);
  bias = arma::mat(1, outSize);
}
int Layer::GetInSize() const {
  return inSize;
}
int Layer::GetOutSize() const {
  return outSize;

}

// TODO: Da testare
//! Ricorda che se vuoi avere run ripetibili, devi usare arma_rng::set_seed(value) al posto di arma::arma_rng::set_seed_random()
void Layer::Init(const double upperBound, const double lowerBound) {
  arma::arma_rng::set_seed_random();
  weight = lowerBound + arma::randu<arma::mat>(inSize, outSize) * (upperBound - lowerBound);
  bias = lowerBound + arma::randu<arma::mat>(1, outSize) * (upperBound - lowerBound);
}
void Layer::Activate(const arma::mat &input, arma::mat &&output) {
  activationFunction.Compute(input, std::move(output));
}
void Layer::SaveOutputParameter(const arma::mat &input) {
  outputParameter = input;
}
void Layer::SaveInputParameter(const arma::mat &input) {
  inputParameter = input;
}
// TODO: Da testare
void Layer::Gradient(const arma::mat &&summationGradientWeight) {

  arma::mat firstDerivativeActivation;
  activationFunction.Derive(std::move(outputParameter), std::move(firstDerivativeActivation));
  firstDerivativeActivation = arma::mean(firstDerivativeActivation);

  //TODO: Pura magia da testare sbagliata
  gradient = firstDerivativeActivation % summationGradientWeight;
  gradient.print("gradient");
}

// TODO: Da testare post backprop
void Layer::AdjustWeight(const double learningRate) {
  if (gradient.n_rows != inputParameter.n_rows) {
    std::cout << "!!!!!!!!!!!!! AdjustWeight errore nelle concordanza colonne - righe !!!!!!!!!!!!!!!!"
              << std::endl;
  }

  weight = weight + learningRate * gradient.t() * inputParameter;

}
/**
 * Return a raw vector contains all the summed weight multiplied by the layer gradient
 */
void Layer::GetSummationWeight(arma::mat &&gradientWeight) {
  if (gradient.n_cols != weight.n_cols) {
    std::cout << "!!!!!!!!!!!!! GetSummationWeight errore nelle concordanza colonne - righe !!!!!!!!!!!!!!!!"
              << std::endl;
  }
  gradientWeight = gradient * weight.t();
}