//
// Created by gs1010 on 01/12/19.
//

#ifndef MLPROJECT_SRC_PREPROCESSING_PREPROCESSING_H_
#define MLPROJECT_SRC_PREPROCESSING_PREPROCESSING_H_

#include "armadillo"
class Preprocessing {
 private:
  arma::mat dataset;
  const std::string datasetPath;

 public:
  explicit Preprocessing(std::string dataset_path);
  void ReturnSplitDataset(int trainPercent,
                          int validationPercent,
                          int testPercent,
                          arma::mat &&trainingSet,
                          arma::mat &&validationSet,
                          arma::mat &&testSet);

// TODO: Funzione di shuffle dei dati (chiamata dall'oggetto network a fine epoca per l'epoca successiva)
};

#endif //MLPROJECT_SRC_PREPROCESSING_PREPROCESSING_H_
