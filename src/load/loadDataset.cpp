//
// Created by gs1010 on 11/12/19.
//

#include "loadDataset.h"

#include <utility>
const arma::mat &LoadDataset::GetDataset() const {
  return dataset;
}
void LoadDataset::Load(std::string path) {
  dataset.load(std::move(path));
}
void LoadDataset::Write(std::string path) {
  dataset.save(path);
}
void LoadDataset::explodeMonkDataset() {
  arma::mat monkDatasetTranslate = arma::zeros<arma::mat>(dataset.n_rows, 18);
  int index[] = {0, 3, 6, 8, 11, 15, 17};
  for (int i = 0; i < dataset.n_rows; i++) {
    for (int y = 0; y < 6; y++) {
      monkDatasetTranslate.at(i, dataset.at(i, y) + index[y] - 1) = 1;
    }
    monkDatasetTranslate.at(i, 17) = dataset.at(i, 6);
  }
  monkDatasetTranslate.save("monks-formatted.csv", arma::csv_ascii);
}