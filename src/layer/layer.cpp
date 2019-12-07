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
  output = weight * input;
  output.each_col() += bias;
}
void Layer::Backward(const arma::mat &&input, arma::mat &&gy, arma::mat &&g) {

}
void Layer::Gradient(const arma::mat &&input, arma::mat &&error, arma::mat &&gradient) {

}
void Layer::Initialize() {
  weight = arma::mat(outSize, inSize);
  bias = arma::mat(outSize, 1);
}
int Layer::GetInSize() const {
  return inSize;
}
int Layer::GetOutSize() const {
  return outSize;
}
void Layer::Init(const double upperBound, const double lowerBound) {
  weight = lowerBound + arma::randu<arma::mat>(outSize, inSize) * (upperBound - lowerBound);
  bias = lowerBound + arma::randu<arma::mat>(outSize, 1) * (upperBound - lowerBound);

  std::cout << weight;
  std::cout << bias;

}
