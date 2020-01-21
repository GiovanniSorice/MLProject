//
// Created by gs1010 on 21/01/20.
//

#include "crossValidation.h"
#include "../network/network.h"
void CrossValidation::run(arma::mat dataset,
                          arma::mat label,
                          int kfold,
                          Network net,
                          int epoch,
                          int batchSize,
                          double learningRate,
                          double weightDecay,
                          double momentum
) {
  arma::mat joinedDataset = arma::join_rows(dataset, label);
  int step = ceil(dataset.n_rows / kfold);
  int inizio = 0;
  int fine = step;
  bool end = false;
  for (int i = 0; i < kfold; i++) {
    std::cout << "inizio " << inizio << " fine " << fine;
    arma::mat validationSet = dataset.submat(inizio, 0, fine,
                                             dataset.n_cols - label.n_cols);
    arma::mat validationLabelSet = dataset.submat(inizio, dataset.n_cols - label.n_cols, fine,
                                                  dataset.n_cols);

    arma::mat firstPartTrainingSet;
    if (inizio != 0) {
      firstPartTrainingSet = dataset.submat(0, 0, inizio - 1,
                                            dataset.n_cols - label.n_cols);
    }

    arma::mat secondPartTrainingSet;
    if (fine != dataset.n_rows - 1) {
      secondPartTrainingSet = dataset.submat(fine + 1, 0, dataset.n_rows,
                                             dataset.n_cols - label.n_cols);
    }

    arma::mat trainingDataset = arma::join_cols(firstPartTrainingSet, secondPartTrainingSet);
    trainingDataset.print("trainingDataset");
    net.Init(1e-4, -1e-4); //todo: maggico
    net.Train(trainingDataset, label.n_cols, epoch, batchSize, learningRate, weightDecay, momentum);

    net.Test(std::move(validationSet), std::move(validationLabelSet));

    inizio = fine + 1;
    fine = (fine + step) < dataset.n_rows ? fine + step : dataset.n_rows - 1;

  }
}
