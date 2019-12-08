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

void Preprocessing::ReturnSplitDataset(int trainPercent,
                                       int validationPercent,
                                       int testPercent,
                                       arma::mat &&trainingSet,
                                       arma::mat &&validationSet,
                                       arma::mat &&testSet) {

  trainingSet = dataset.submat(0, 0, std::floor(dataset.n_rows * trainPercent / 100),
                               dataset.n_cols - 1);

  validationSet = dataset.submat(trainingSet.n_rows,
                                 0,
                                 trainingSet.n_rows
                                     + std::floor(dataset.n_rows * validationPercent / 100),
                                 dataset.n_cols - 1);

  testSet = dataset.submat(
      trainingSet.n_rows + validationSet.n_rows,
      0,
      dataset.n_rows - 1,
      dataset.n_cols - 1);
}