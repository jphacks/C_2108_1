mkdir tmp
cd ./tmp

wget -c https://github.com/ku-nlp/jumanpp/releases/download/v2.0.0-rc3/jumanpp-2.0.0-rc3.tar.xz
tar xf jumanpp-2.0.0-rc3.tar.xz
mkdir bld
cd bld
cmake ../jumanpp-2.0.0-rc3 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/usr/local
sudo make install

cd ../../
rm -r ./tmp