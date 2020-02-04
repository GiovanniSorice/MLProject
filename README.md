

<img src="https://github.com/GiovanniSorice/MLProject/blob/master/logo/neuradillo.jpg" height="200" width="300">


# Neuradillo
Neuradillo is a small C++ exploiting [Armadillo](http://arma.sourceforge.net/) numerical library to create and train feedforward neural network. We developed this library as a project for the Machine Learning course. More information about the project can be found in the [Report](https://github.com/GiovanniSorice/MLProject/blob/master/docs/report/relazione.pdf). 

## Getting started

### Prerequisites 
The project use [Cmake 3.16](https://cmake.org/) as building system and it can be downloaded [here](https://cmake.org/download/). 
The package manager used is [Conan](https://conan.io/). You can install it through Pip with the following commands:
` pip install conan` 

Running the following command to solve this [issue](https://docs.conan.io/en/latest/installation.html#install-with-pip-recommended): 
`source ~/.profile`  

### Armadillo installation 
1. Clone the following repo: https://github.com/darcamo/conan-armadillo;
2. Inside the cloned repo run: `conan create . armadillo/stable`
3. If Armadillo is installed correctly an example program is execute and you can start use it through Conan.

## Running the project
2. Eseguire il progetto: 
    1. Aprire una shell nella cartella del progetto e spostarsi all'interno della cartella MLProject/ ; 
    2. All'interno della cartella dare il seguente comando da shell: conan install . -s build_type=Release --install-folder=build
    3. Se il comando è avvenuto correttamente una cartella "build" viene creata con all'interno i file di cofigurazione di Cmake.
    4. Nella stessa cartella dare il seguente comando da shell:  cmake -Bbuild -H. 
    5. Spostarsi nella cartella generata dando: cd build/
    6. Dare da shell: cmake --build . --config Release 
    7. Spostarsi nella cartella bin/ ; 
    8. Eseguire il sorgente ottenuto dando: ./MLProject

## Example 
Show CUP neural network setup and train.

## Future works
- boost serialization: https://www.boost.org/doc/libs/1_72_0/libs/serialization/doc/index.html 
- matplotlibcpp: https://github.com/lava/matplotlib-cpp 

## Authors
* **Giovanni Sorice**  :computer: - [Giovanni Sorice](https://github.com/GiovanniSorice)
* **Francesco Corti** :nerd_face: :computer: - [FraCorti](https://github.com/FraCorti)