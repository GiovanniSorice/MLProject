//
// Created by gs1010 on 01/12/19.
//

#include "preprocessing.h"

#include <utility>
preprocessing::preprocessing(std::string dataset_path)
    : datasetPath(std::move(dataset_path)), trainPercent(60),
      validationPercent(20),
      testPercent(20) {
  dataset.load(datasetPath, arma::csv_ascii);
  std::cout << dataset;

}
preprocessing::preprocessing(std::string dataset_path,
                             int train_percent,
                             int validation_percent,
                             int test_percent)
    : datasetPath(std::move(dataset_path)),
      trainPercent(train_percent),
      validationPercent(validation_percent),
      testPercent(test_percent) {
  dataset.load(datasetPath, arma::csv_ascii);
}
const arma::Mat<double> *preprocessing::GetTrainingSet() const {
  return new arma::mat(dataset.submat(0, 0, dataset.n_rows * trainPercent / 100,
                                      dataset.n_cols - 1));
}
const arma::Mat<double> *preprocessing::GetValidationSet() const {
  return new arma::mat(dataset.submat(dataset.n_rows * trainPercent / 100 + 1,
                                      0,
                                      dataset.n_rows * trainPercent / 100 + 1
                                          + dataset.n_rows * validationPercent / 100,
                                      dataset.n_cols - 1));
}
const arma::Mat<double> *preprocessing::GetTestSet() const {
  return new arma::mat(dataset.submat(
      dataset.n_rows * trainPercent / 100 + dataset.n_rows * validationPercent / 100 + 1,
      0,
      dataset.n_rows - 1,
      dataset.n_cols - 1));
}
int preprocessing::GetTrainPercent() const {
  return trainPercent;
}
void preprocessing::SetTrainPercent(int train_percent) {
  trainPercent = train_percent;
}
int preprocessing::GetValidationPercent() const {
  return validationPercent;
}
void preprocessing::SetValidationPercent(int validation_percent) {
  validationPercent = validation_percent;
}
int preprocessing::GetTestPercent() const {
  return testPercent;
}
void preprocessing::SetTestPercent(int test_percent) {
  testPercent = test_percent;
}
