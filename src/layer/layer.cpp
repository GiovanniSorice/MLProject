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
  return deltaWeight;
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
const arma::mat &Layer::GetDeltaBias() const {
  return deltaBias;
}

Layer::Layer(const int inSize, const int outSize, ActivationFunction &activationFunction)
    : inSize(inSize),
      outSize(outSize),
      activationFunction(activationFunction),
      deltaWeight(arma::zeros(inSize, outSize)),
      deltaBias(arma::zeros(1, outSize)) {
}

/** Given the activated vector of the previous layer compute the forward pass
 *
 *  @param input Previous activated vector
 *  @param output Forwarded vector computed through weight and bias of the current layer
 * */
void Layer::Forward(const arma::mat &&input, arma::mat &&output) {
  //input.print("Input");
  //weight.print("Weight");
  //bias.print("bias");
  output = weight * input;
  output.each_col() += bias;
  //output.print("output");

}
void Layer::Backward(const arma::mat &&input, arma::mat &&gy, arma::mat &&g) {

}

/** Compute the gradient of the output layer
 *
 *  @param partialDerivativeOutput Partial derivative of the output neuron
 * */
void Layer::OutputLayerGradient(const arma::mat &&partialDerivativeOutput) {
  arma::mat firstDerivativeActivation;
  //outputParameter.print("outputParameter");
  activationFunction.Derive(std::move(outputParameter), std::move(firstDerivativeActivation));
  firstDerivativeActivation = arma::sum(firstDerivativeActivation, 1);
  //firstDerivativeActivation.print("First derivative activation");
  //partialDerivativeOutput.print("partialDerivativeOutput");
  gradient = partialDerivativeOutput % firstDerivativeActivation;
  //gradient.print("Output Layer gradient");
  // (partialDerivativeOutput % firstDerivativeActivation).print("partialDerivativeOutput % firstDerivativeActivation");
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
/***/
void Layer::Init(const double upperBound, const double lowerBound) {
  //arma::arma_rng::set_seed(9);
  arma::arma_rng::set_seed_random();
  weight = lowerBound + arma::randu<arma::mat>(outSize, inSize) * (upperBound - lowerBound);
  bias = lowerBound + arma::randu<arma::mat>(outSize, 1) * (upperBound - lowerBound);
  //bias = arma::zeros<arma::mat>(1, outSize);

}
void Layer::Activate(const arma::mat &input, arma::mat &&output) {
  activationFunction.Compute(input, std::move(output));
  //input.print("input");
  //output.print("activated output");
}
void Layer::SaveOutputParameter(const arma::mat &input) {
  outputParameter = input;
}
void Layer::SaveInputParameter(const arma::mat &input) {
  inputParameter = input;
}
/***/
void Layer::Gradient(const arma::mat &&summationGradientWeight) {

  arma::mat firstDerivativeActivation;
  activationFunction.Derive(std::move(outputParameter), std::move(firstDerivativeActivation));
  firstDerivativeActivation = arma::sum(firstDerivativeActivation, 1);

  gradient = firstDerivativeActivation % summationGradientWeight;
  //gradient.print("Hidden layer gradient");
}

/***/
void Layer::AdjustWeight(const double learningRate, const double weightDecay, const double momentum) {
  weight = weight + momentum * deltaWeight - learningRate * gradient * arma::sum(inputParameter, 1).t()
      - 2 * weightDecay * weight;
  bias = bias + momentum * deltaBias - learningRate * gradient - 2 * weightDecay * bias;
  deltaWeight = momentum * deltaWeight - learningRate * gradient * arma::sum(inputParameter, 1).t();
  deltaBias = momentum * deltaBias - learningRate * gradient;
}

/**
 * Return a raw vector contains all the summed weight multiplied by the layer gradient
 */
void Layer::GetSummationWeight(arma::mat &&gradientWeight) {
  gradientWeight = weight.t() * gradient;
}
