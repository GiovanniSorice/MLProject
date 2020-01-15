//
// Created by gs1010 on 11/12/19.
//

#ifndef MLPROJECT_SRC_LOADDATASET_H_
#define MLPROJECT_SRC_LOADDATASET_H_

#include <string>
#include <armadillo>
class LoadDataset {
  arma::mat dataset;
 public:
  void Load(std::string);
  void Write(std::string);
  void explodeMonkDataset();
  [[nodiscard]] const arma::mat &GetDataset() const;
};

#endif //MLPROJECT_SRC_LOADDATASET_H_
