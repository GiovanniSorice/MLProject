//
// Created by checco on 26/11/19.
//

#ifndef MLPROJECT_SRC_LAYER_H_
#define MLPROJECT_SRC_LAYER_H_

#include "armadillo"
#include "../activationFunction/activationFunction.h"

class Layer {
 private:
  int inSize;
  int outSize;
  //! weights pesi del layer corrente poer ogni nodo al suo interno
  arma::mat weight;
  //! bias del layer corrente
  arma::mat bias;
  //! locally-instored delta object
  arma::mat delta;
  //! gradiente del layer
  arma::mat gradient;
  //! parametri di input del layer
  arma::mat inputParameter;
  //! parametri di output del layer
  arma::mat outputParameter;
  //! ActivationFunction utilizzata nel layer
  ActivationFunction &activationFunction;
 public:
  /**
   * Create the Linear layer object using the specified number of units.
   *
   * @param inSize The number of input units.
   * @param outSize The number of output units.
   */
  Layer(const int inSize, const int outSize, ActivationFunction &activationFunction);

  /**
   * Initialize the layer parameter.
   */
  void Initialize();
  /**
   *
   * @param input
   * @param output
   */
  void Activate(const arma::mat &input, arma::mat &&output);
  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */

  void Forward(const arma::mat &&input, arma::mat &&output);
  /**
  * Ordinary feed backward pass of a neural network, calculating the function
  * f(x) by propagating x backwards trough f. Using the results from the feed
  * forward pass.
  *
  * @param input The propagated input activation.
  * @param gy The backpropagated error.
  * @param g The calculated gradient.
  */
  void Backward(const arma::mat &&input, arma::mat &&gy, arma::mat &&g);

  /**
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The input parameter used for calculating the gradient.
   *
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  void OutputLayerGradient(const arma::mat &&error);
  void Gradient(const arma::mat &&summationGradientWeight);
  void SaveOutputParameter(const arma::mat &input);
  void SaveInputParameter(const arma::mat &input);
  void AdjustWeight(const double learningRate);
  [[nodiscard]] const arma::mat &GetWeight() const;
  [[nodiscard]] const arma::mat &GetBias() const;
  [[nodiscard]] const arma::mat &GetDelta() const;
  [[nodiscard]] const arma::mat &GetGradient() const;
  [[nodiscard]] const arma::mat &GetInputParameter() const;
  [[nodiscard]] const arma::mat &GetOutputParameter() const;
  [[nodiscard]] int GetInSize() const;
  [[nodiscard]] int GetOutSize() const;
  void Init(const double upperBound, const double lowerBound);
// Costruttore:
//  - con funzione di attivazione;
//  - Unità in input e output;
// Metodi:
//  - Forward;
//  - Backward;
//  - Gradient;
//  - BackProp??
//  - Altre funzioni di apprendimento?
};

#endif //MLPROJECT_SRC_LAYER_H_

