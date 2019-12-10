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

void Layer::Forward(const arma::mat &&input, arma::mat &&output) {
  output = input * weight;
  output.each_row() += bias;
}
void Layer::Backward(const arma::mat &&input, arma::mat &&gy, arma::mat &&g) {

}
void Layer::OutputLayerGradient(const arma::mat &&input, const arma::mat &&error, arma::mat &&gradient) {
  arma::mat firstDerivativeActivation;
  activationFunction.Derive(std::move(input), std::move(firstDerivativeActivation));
  firstDerivativeActivation.print("firstDerivativeActivation");
  error.print("error");
  gradient = arma::mean(error * firstDerivativeActivation);
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
void Layer::Gradient(const arma::mat &&summationGradientWeight, arma::mat &&currentGradientWeight) {

  arma::mat firstDerivativeActivation;
  activationFunction.Derive(std::move(outputParameter), std::move(firstDerivativeActivation));
  firstDerivativeActivation.print("firstDerivativeActivation");
  summationGradientWeight.print("summationGradientWeight");
  gradient = arma::mean(firstDerivativeActivation * summationGradientWeight);
  gradient.print("gradient");
  currentGradientWeight = gradient * weight.t();
}
