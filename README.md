#Neuradillo

1. Install python 3 e dare: pip3 install conan, dopo dare un: source ~/.profile  
2. Installare armadillo dala repo di Darcamo:  https://github.com/darcamo/conan-armadillo 
3. 
4. Eseguire il progetto senza CLion: 
    1. Dare: conan install . -s build_type=Release --install-folder=cmake-build-release
    2. Dare cmake . // da capire se si può mettere in una folder, forse: cmake -Bcmake-build-release -H.
    3. Spostarsi nella cartella generata dando: cd cd cmake-build-release/
    4. Dall'interno della tabella dare: cmake --build . --config Release 
    5. Spostarsi nella cartella bin/ dando: cd bin
    6. Eseguire il sorgente con: ./MLProject 