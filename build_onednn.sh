

mkdir -p thirdparty/oneDNN/build
cd thirdparty/oneDNN/build
cmake -DCMAKE_INSTALL_PREFIX=./install .
cmake --build . --target install
