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
                          double momentum,
                          arma::mat meanError
) {
  arma::mat joinedDataset = arma::join_rows(dataset, label);
  int step = ceil(dataset.n_rows / kfold);
  int start = 0;
  int end = step;

  arma::mat currentError = arma::zeros(1, label.n_cols);

  for (int i = 0; i < kfold; i++) {
    std::cout << "start " << start << " fine " << end - 1;
    arma::mat validationSet = dataset.submat(start, 0, end - 1,
                                             dataset.n_cols - 1);
    arma::mat validationLabelSet = label.submat(start, 0, end - 1,
                                                label.n_cols - 1);

    arma::mat firstPartTrainingSet;
    if (start != 0) {
      firstPartTrainingSet = joinedDataset.submat(0, 0, start - 1,
                                                  joinedDataset.n_cols - 1);
    }

    arma::mat secondPartTrainingSet;
    if (end != dataset.n_rows - 1) {
      secondPartTrainingSet = joinedDataset.submat(end + 1, 0, joinedDataset.n_rows,
                                                   joinedDataset.n_cols - 1);
    }

    arma::mat trainingDataset = arma::join_cols(firstPartTrainingSet, secondPartTrainingSet);
    net.Clear();
    net.Init(1e-4, -1e-4); //todo: maggico
    net.Train(trainingDataset, label.n_cols, epoch, batchSize, learningRate, weightDecay, momentum);

    net.Test(std::move(validationSet), std::move(validationLabelSet), std::move(currentError));
    meanError += meanError;
    start = end;
    end = (end + step) < dataset.n_rows ? end + step : dataset.n_rows - 1;

  }
  meanError /= kfold;
}
