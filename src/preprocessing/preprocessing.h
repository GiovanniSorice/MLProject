//
// Created by gs1010 on 01/12/19.
//

#ifndef MLPROJECT_SRC_PREPROCESSING_PREPROCESSING_H_
#define MLPROJECT_SRC_PREPROCESSING_PREPROCESSING_H_

#include "armadillo"
class Preprocessing {
 private:
  arma::Mat<double> dataset;
  arma::Mat<double> testSet;
  arma::Mat<double> validationSet;
  arma::Mat<double> trainingSet;
  const std::string datasetPath;
  int trainPercent;
  int validationPercent;
  int testPercent;

 public:
  explicit Preprocessing(std::string dataset_path);
  Preprocessing(std::string dataset_path, int train_percent, int validation_percent, int test_percent);
  [[nodiscard]] int GetTrainPercent() const;
  void SetTrainPercent(int train_percent);
  [[nodiscard]] int GetValidationPercent() const;
  void SetValidationPercent(int validation_percent);
  [[nodiscard]] int GetTestPercent() const;
  void SetTestPercent(int test_percent);
  const arma::Mat<double> &GetTestSet() const;
  const arma::Mat<double> &GetValidationSet() const;
  const arma::Mat<double> &GetTrainingSet() const;

  //Costruttore:
  // - Percorso file
  // - % di train
  // - % di validation
  // - % di test
  //METODI
// Legge il file
// Suddivide il dataset in train, validation e test set
// Funzione di shuffle dei dati (chiamata dall'oggetto network a fine epoca per l'epoca successiva)

};

#endif //MLPROJECT_SRC_PREPROCESSING_PREPROCESSING_H_
