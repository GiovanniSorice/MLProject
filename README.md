#Neuradillo

1. Installazione del package manager Conan: 
   1. Assicurarsi di avere installato python 3: 
   2. Dopo aver installato python3 correttamente installare il package manager Conan (https://conan.io/) dando da shell il seguente comando: pip3 install conan
   3. L'installazione di conan tramite pip potrebbe non essere rilevata dal sistema, dare il comando da shell: source ~/.profile
   4. Eseguire:  pip install conan --upgrade per aggiornare Conan 

2. Installazione di Armadillo: 
   1. Clonare la seguente repo dando il seguente comando da shell: git clone  https://github.com/darcamo/conan-armadillo 
   2. Spostarsi all'interno della repo, fare particolare attenzione a mettersi nella cartella contenente il file Conafile.py
   3. Dare il comando: conan create . armadillo/stable
   4. Se tutto avviene correttamente la libreria viene installata e viene eseguito un programma di esempio; 


3. Eseguire il progetto: 
    1. Aprire una shell nella cartella del progetto e spostarsi all'interno della cartella MLProject/ ; 
    2. All'interno della cartella dare il seguente comando da shell: conan install . -s build_type=Release --install-folder=build
    3. Se il comando è avvenuto correttamente una cartella "build" viene creata con all'interno i file di cofigurazione di Cmake.
    4. Nella stessa cartella dare il seguente comando da shell:  cmake -Bbuild -H. 
    5. Spostarsi nella cartella generata dando: cd build/
    6. Dare da shell: cmake --build . --config Release 
    7. Spostarsi nella cartella bin/ ; 
    8. Eseguire il sorgente ottenuto dando: ./MLProject