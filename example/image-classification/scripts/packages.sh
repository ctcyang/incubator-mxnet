echo $SHELL
pwd
echo $LD_LIBRARY_PATH
which pip
sudo dpkg -i nccl-repo-ubuntu1604-2.1.15-ga-cuda9.0_1-1_amd64.deb
sudo apt-key add /var/nccl-repo-2.1.15-ga-cuda9.0/7fa2af80.pub
sudo apt update
sudo apt install libnccl2 libnccl-dev

sudo /home/ubuntu/anaconda3/bin/pip install tensorflow-gpu
sudo LD_LIBRARY_PATH=/usr/lib64/openmpi/lib/:/usr/local/cuda/lib64:/usr/local/lib:/usr/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/mpi/lib:/lib/:/home/ubuntu/src/cntk/bindings/python/cntk/libs:/usr/local/cuda/lib64:/usr/local/lib:/usr/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/mpi/lib:/usr/lib64/openmpi/lib/:/usr/local/cuda/lib64:/usr/local/lib:/usr/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/mpi/lib:/lib/ HOROVOD_NCCL_HOME=/usr/lib/x86_64-linux-gnu HOROVOD_GPU_ALLREDUCE=NCCL /home/ubuntu/anaconda3/bin/pip install --no-cache-dir horovod
