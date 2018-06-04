sudo apt-get install -y autoconf automake libtool nasm && \
JPEG_TURBO_VERSION=1.5.2 && \
wget -q -O - https://github.com/libjpeg-turbo/libjpeg-turbo/archive/${JPEG_TURBO_VERSION}.tar.gz | tar -xzf - && \
cd libjpeg-turbo-${JPEG_TURBO_VERSION} && \
autoreconf -fiv && \
./configure --enable-shared --prefix=/usr 2>&1 >/dev/null && \
sudo make -j"$(nproc)" install 2>&1 >/dev/null && \
rm -rf libjpeg-turbo-${JPEG_TURBO_VERSION}
cd ~
git clone --recursive https://github.com/rahul003/mxnet --branch rec-aug mxnet
git clone --recursive https://github.com/dmlc/gluon-cv
cd ~/gluon-cv
git apply --verbose ~/mxnet/example/image-classification/gamma-patch-gluoncv.diff
sudo /home/ubuntu/anaconda3/bin/python setup.py install

cd ~/mxnet
git pull
make -j 64
cd python/
sudo /home/ubuntu/anaconda3/bin/python setup.py install
