//
// Created by gs1010 on 01/12/19.
//

#include "preprocessing.h"

#include <utility>
Preprocessing::Preprocessing(std::string
                             dataset_path)
    : datasetPath(std::move(dataset_path)) {
  dataset.load(datasetPath, arma::csv_ascii);
}

void Preprocessing::GetSplit(int trainPercent,
                             int validationPercent,
                             int testPercent,
                             arma::mat &&trainingSet,
                             arma::mat &&validationSet,
                             arma::mat &&testSet) {
  dataset = arma::shuffle(dataset);
  //TODO: va bene qui lo shiuffle?

  if (trainPercent) {
    trainingSet = dataset.submat(0, 0, std::floor(dataset.n_rows * trainPercent / 100) - 1,
                                 dataset.n_cols - 1);
  }

  if (validationPercent) {
    validationSet = dataset.submat(trainingSet.n_rows,
                                   0,
                                   trainingSet.n_rows
                                       + std::floor(dataset.n_rows * validationPercent / 100) - 1,
                                   dataset.n_cols - 1);
  }
  if (testPercent) {
    testSet = dataset.submat(
        trainingSet.n_rows + validationSet.n_rows,
        0,
        dataset.n_rows - 1,
        dataset.n_cols - 1);
  }
}

void Preprocessing::GetTrainingSet(int trainPercent,
                                   int validationPercent,
                                   int testPercent,
                                   arma::mat &&trainingSet) {
  trainingSet = dataset.submat(0, 0, std::floor(dataset.n_rows * trainPercent / 100) - 1,
                               dataset.n_cols - 1);

}

void Preprocessing::GetValidationSet(int trainPercent,
                                     int validationPercent,
                                     int testPercent,
                                     arma::mat &&validationSet) {
  validationSet = dataset.submat(std::floor(dataset.n_rows * trainPercent / 100) + 1,
                                 0,
                                 std::floor(dataset.n_rows * trainPercent / 100)
                                     + std::floor(dataset.n_rows * validationPercent / 100),
                                 dataset.n_cols - 1);
}
void Preprocessing::GetTestSet(int trainPercent,
                               int validationPercent,
                               int testPercent,
                               arma::mat &&testSet) {
  testSet = dataset.submat(
      std::floor(dataset.n_rows * trainPercent / 100) + std::floor(dataset.n_rows * validationPercent / 100) + 1,
      0,
      dataset.n_rows - 1,
      dataset.n_cols - 1);
}