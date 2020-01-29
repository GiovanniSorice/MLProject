#Neuradillo

1. Assicurarsi di avere installato python 3 nella propria macchina
   1. Dopo aver installato python3 correttamente installare il package manager Conan (https://conan.io/) dando da shell il seguente comando: pip3 install conan
   2. L'installazione di conan tramite pip potrebbe non essere rilevata dal sistema, dare il comando da shell: source ~/.profile
   3. Eseguire:  pip install conan --upgrade per aggiornare Conan 

2. Clonare la seguente repo dando il seguente comando da shell: git clone  https://github.com/darcamo/conan-armadillo 
   1. Spostarsi all'interno della repo, fare particolare attenzione a mettersi nella cartella contenente il Conafile.py
   2. Dare il comando: conan create . armadillo/stable
   3. Se tutto avviene correttamente il programma stampa un "Hello world" di test; 


3. Eseguire il progetto da shell dopo l'installazione di Conan: 
    1. Aprire una shell nella cartella del progetto e spostarsi all'interno della cartella MLProject/ ; 
    2. All'interno della cartella dare il seguente comando da shell: conan install . -s build_type=Release --install-folder=build
    3. Nella stessa cartella dare il seguente comando da shell:  cmake -Bbuild -H. 
    4. Spostarsi nella cartella generata dando: cd build/
    5. Dare da shell: cmake --build . --config Release 
    6. Spostarsi nella cartella bin/ ; 
    7. Eseguire il sorgente ottenuto dando: ./MLProject 