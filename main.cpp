#include <iostream>
#include "armadillo"
#include "src/preprocessing/preprocessing.h"

int main() {
  arma::mat matrix1;
  //matrix1.load("../../data/ML-CUP19-TR.csv", arma::csv_ascii);
  //arma::mat trainData = matrix1.submat(0, 0, matrix1.n_rows - 4,
  //matrix1.n_cols - 1);
  //std::cout << "trainData" << trainData << std::endl;

  preprocessing a("/home/gs1010/CLionProjects/MLProject/data/monk/monk_dataset.csv", 60, 20, 20);

  std::cout << " TrainingPercent " << a.GetTrainPercent() << " ValidationPercent " << a.GetValidationPercent()
            << " TestPercent " << a.GetTestPercent() << std::endl;
  std::cout << " TestSet" << *a.GetTrainingSet();
  /*
  // Load the training set.
  arma::mat dataset;
  dataset.load("../../data/monk/monk_dataset.csv", arma::csv_ascii);
  //std::cout << "trainData" <<std::endl<< dataset << std::endl;

// Split the labels from the training set.
  arma::mat trainData = dataset.submat(0, 0, dataset.n_rows - 4,
                                       dataset.n_cols - 1);
  std::cout << "trainData" << std::endl << &dataset << " " << &trainData << std::endl;

// Split the data from the training set.
  arma::mat trainLabelsTemp = dataset.submat(dataset.n_rows - 3, 0,
                                             dataset.n_rows - 1, dataset.n_cols - 1);
  std::cout << "trainLabelsTemp" << std::endl << trainLabelsTemp << std::endl;
*/
  return 0;
}

