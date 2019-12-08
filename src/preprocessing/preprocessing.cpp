//
// Created by gs1010 on 01/12/19.
//

#include "preprocessing.h"

#include <utility>
Preprocessing::Preprocessing(std::string
                             dataset_path)
    : datasetPath(std::move(dataset_path)), trainPercent(60),
      validationPercent(20),
      testPercent(20) {
  dataset.load(datasetPath, arma::csv_ascii);
  std::cout << dataset.n_elem << "elementi dataset" << std::endl;

}
Preprocessing::Preprocessing(std::string dataset_path,
                             int train_percent,
                             int validation_percent,
                             int test_percent)
    : datasetPath(std::move(dataset_path)),
      trainPercent(train_percent),
      validationPercent(validation_percent),
      testPercent(test_percent) {
  dataset.load(datasetPath, arma::csv_ascii);

  std::cout << dataset.n_elem << "elementi dataset" << std::endl;

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

  /*testTrainingSubMat.impl_print("SubMat trainingSet");
  trainingSet =
      arma::mat(dataset.memptr(), dataset.n_rows * 0.6, dataset.n_cols, false, false);
  validationSet =
      arma::mat(dataset.memptr() + trainingSet.n_elem,
                std::floor(dataset.n_rows * validation_percent / 100),
                dataset.n_cols,
                false,
                false);
  testSet =
      arma::mat(dataset.memptr() + trainingSet.n_elem + validationSet.n_elem,
                std::ceil(dataset.n_rows - trainingSet.n_rows - validationSet.n_rows),
                dataset.n_cols,
                false,
                false); */

}
int Preprocessing::GetTrainPercent() const {
  return trainPercent;
}
void Preprocessing::SetTrainPercent(int train_percent) {
  trainPercent = train_percent;
}
int Preprocessing::GetValidationPercent() const {
  return validationPercent;
}
void Preprocessing::SetValidationPercent(int validation_percent) {
  validationPercent = validation_percent;
}
int Preprocessing::GetTestPercent() const {
  return testPercent;
}
void Preprocessing::SetTestPercent(int test_percent) {
  testPercent = test_percent;
}
const arma::mat &Preprocessing::GetTestSet() const {
  return testSet;
}
const arma::mat &Preprocessing::GetValidationSet() const {
  return validationSet;
}
const arma::mat &Preprocessing::GetTrainingSet() const {
  return trainingSet;
}
