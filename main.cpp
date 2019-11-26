#include <iostream>
#include "armadillo"


int main() {
    std::cout << "Hello, World!" << std::endl;
    std::cout << "Armadillo version: " << arma::arma_version::as_string() << std::endl;
    arma::arma_rng::set_seed_random();
    arma::Mat<double > A = arma::randu(4,4);
    std::cout << "A:\n" << A << "\n";
    std::cout <<"Hello world Conan with Armadillo in src" << std::endl;
    // arma::mat
    return 0;
}