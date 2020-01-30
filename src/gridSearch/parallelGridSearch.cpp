//
// Created by checco on 25/01/20.
//
#include <chrono>
#include <thread>
#include "parallelGridSearch.h"
#include "gridSearch.h"

/**  Given the dataset and the label the computation is divided among multiple thread, all the
 *   result are stored inside std::vector result matrix and then retrieved
 *
 *   @param dataset The dataset used for training
 *   @param label Label used for training the network
 * */
void ParallelGridSearch::Run(arma::mat dataset, arma::mat label) {
  int parallelThreads = std::thread::hardware_concurrency() / 1;

  // save number of threads wanted
  setNumberThread(parallelThreads);

  int totalNetworkAnalyzed = NetworkAnalyzed();
  // set up grid search vector
  setGridsSearch(totalNetworkAnalyzed);
  auto parallelGridIterator = gridSearches.begin();

  // start all the grid search in parallel
  for (arma::mat &matrix : resultMatrix) {
    gridSearchThreads.emplace_back(std::thread(&GridSearch::Run,
                                               &(*parallelGridIterator),
                                               dataset,
                                               label,
                                               std::move(matrix)));
    parallelGridIterator++;
  }

  // wait all the threads to stop
  for (std::thread &thread :gridSearchThreads) {
    if (thread.joinable()) {
      thread.join();
    }
  }
  // save result of the grid searches
  saveResult();
}

/**  Threads computed results are joined and saved in a unique file
 * */
void ParallelGridSearch::saveResult() {
  arma::mat result;
  // join matrices for saving
  for (arma::mat &matrix: resultMatrix) {
    matrix.print("current result matrix");
    result = arma::join_vert(result, matrix);
  }
  result.save("parallel-grid-search-values.txt", arma::arma_ascii);
}

/** Parameters of all the grid search are set
 *
 * @param totalNetworkAnalyzed The total number of the network computed by the grid search
 * */
void ParallelGridSearch::setGridsSearch(int totalNetworkAnalyzed) {
  int currentEpochInterval = epochInterval();
  int currentUnitInterval = unitInterval();
  double currentMomentumInterval = momentumInterval();
  double currentLambdaInterval = lambdaInterval();
  double currentLearningRateInterval = learningRateInterval();


  // fill the vector with the matrix and create grid search object
  for (int currentGridSearch = 0; currentGridSearch < totalThreadNumber; currentGridSearch++) {
    arma::mat currentResultMatrix = arma::zeros(ceil(totalNetworkAnalyzed / totalThreadNumber), 8);
    resultMatrix.emplace_back(currentResultMatrix);

    GridSearch gridSearch;
    gridSearch.SetEpoch(epochMin, epochMax, epochStep);
    gridSearch.SetLambda(lambdaMin, lambdaMax, lambdaStep);
    gridSearch.SetLearningRate(learningRateMin + currentLearningRateInterval * currentGridSearch,
                               learningRateMin + currentLearningRateInterval * (currentGridSearch + 1),
                               learningRateStep);
    gridSearch.SetMomentum(momentumMin, momentumMax, momentumStep);
    gridSearch.SetUnit(unitMin, unitMax, unitStep);
    gridSearches.push_back(gridSearch);
  }
}

double ParallelGridSearch::learningRateInterval() {
  double interval = (learningRateMax - learningRateMin) / totalThreadNumber;
  return interval;
}

double ParallelGridSearch::momentumInterval() {
  double interval = (momentumMax - momentumMin) / totalThreadNumber;
  return 0;
}

double ParallelGridSearch::lambdaInterval() {
  double interval = (lambdaMax - lambdaMin) / totalThreadNumber;
  return 0;
}

int ParallelGridSearch::epochInterval() {
  int interval = ceil((double) (epochMax - epochMin) / totalThreadNumber);
  return 0;
}

int ParallelGridSearch::unitInterval() {
  int interval = ceil((double) (unitMax - unitMin) / totalThreadNumber);
  return 0;
}

void ParallelGridSearch::setNumberThread(int totalThread) {
  totalThreadNumber = totalThread;
}

/** A
 * */
int ParallelGridSearch::NetworkAnalyzed() {
  double epochN = (epochMax - epochMin) / epochStep + 1;
  double lambdaN = (lambdaMax - lambdaMin) / lambdaStep + 1;
  double lRN = (learningRateMax - learningRateMin) / learningRateStep + 1;
  double momentumN = (momentumMax - momentumMin) / momentumStep + 1;
  double unitN = (unitMax - unitMin) / unitStep + 1;

  int networkAnalyzed = ceil(epochN * lambdaN * lRN * momentumN * unitN);
  return networkAnalyzed;
}
void ParallelGridSearch::SetUnitStep(int unit_step) {
  unitStep = unit_step;
}
void ParallelGridSearch::SetLearningRateStep(double learning_rate_step) {
  learningRateStep = learning_rate_step;
}
void ParallelGridSearch::SetLambdaStep(double lambda_step) {
  lambdaStep = lambda_step;
}
void ParallelGridSearch::SetMomentumStep(double momentum_step) {
  momentumStep = momentum_step;
}
void ParallelGridSearch::SetEpochStep(int epoch_step) {
  epochStep = epoch_step;
}
void ParallelGridSearch::SetLearningRateMin(double learning_rate_min) {
  learningRateMin = learning_rate_min;
}
void ParallelGridSearch::SetLearningRateMax(double learning_rate_max) {
  learningRateMax = learning_rate_max;
}
void ParallelGridSearch::SetLambdaMin(double lambda_min) {
  lambdaMin = lambda_min;
}
void ParallelGridSearch::SetLambdaMax(double lambda_max) {
  lambdaMax = lambda_max;
}
void ParallelGridSearch::SetMomentumMin(double momentum_min) {
  momentumMin = momentum_min;
}
void ParallelGridSearch::SetMomentumMax(double momentum_max) {
  momentumMax = momentum_max;
}
void ParallelGridSearch::SetUnitMin(int unit_min) {
  unitMin = unit_min;
}
void ParallelGridSearch::SetUnitMax(int unit_max) {
  unitMax = unit_max;
}
void ParallelGridSearch::SetEpochMin(int epoch_min) {
  epochMin = epoch_min;
}
void ParallelGridSearch::SetEpochMax(int epoch_max) {
  epochMax = epoch_max;
}
