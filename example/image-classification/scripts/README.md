To install Horovod on top of MXNet:

# Environment
1) Ubuntu 16.04
2) NVCC (we tried with both CUDA 9.0 and 9.2)
3) CUDA Driver (we tried with both 384.111 and 396.37)
4) GCC 5.4.0
5) Install all dependencies following steps for [standard MXNet install](https://mxnet.incubator.apache.org/install/index.html?platform=Linux&language=Python&processor=GPU&version=master#)

# Additional dependencies required compared to vanilla MXNet
1) MPI (we tested using OpenMPI 3.1.1 compiled without CUDA-aware)
2) NCCL (we tested with both NCCL 2.1 and 2.2)

# Building MXNet
1) git clone --recursive https://github.com/ctcyang/incubator-mxnet-1.git -b horovod_feature mxnet
2) Verify the branch is on **horovod_feature**
3) cd mxnet & cp make/config.mk .
4) Only config.mk + Makefile build chain works. We have not added CMakeLists support yet. Make following changes to config.mk. Note: these are *in addition* to the standard `USE_CUDA = 1` and `USE_CUDA_PATH = [cuda directory]` additions to config.mk when building MXNet for GPU:
  ```
  USE_NCCL = 1
  USE_NCCL_PATH = [directory in which libnccl.so resides]
  ```
5) make -j16
6) pip install -e python
7) mkdir python/mxnet/include
8) Enter the following commands 
```
mkdir -p ~/mxnet/python/mxnet/include/mshadow
cp -r ~/mxnet/3rdparty/mshadow/mshadow/* ~/mxnet/python/mxnet/include/mshadow
cp -r ~/mxnet/3rdparty/dlpack/include/dlpack ~/mxnet/python/mxnet/include
cp -r ~/mxnet/3rdparty/dmlc-core/include/dmlc ~/mxnet/python/mxnet/include
cp -r ~/mxnet/3rdparty/tvm/include/tvm ~/mxnet/python/mxnet/include
cp -r ~/mxnet/3rdparty/tvm/nnvm/include/nnvm ~/mxnet/python/mxnet/include
cp -r ~/mxnet/include/mxnet ~/mxnet/python/mxnet/include
```

# Building Horovod
1) git clone https://github.com/ctcyang/horovod.git -b mxnet_feature_fp16 horovod
2) Verify the branch is on **mxnet_feature_fp16**
3) cd horovod
4) sudo PATH=/usr/local/bin:$PATH LD_LIBRARY_PATH=$LD_LIBRARY_PATH HOROVOD_NCCL_HOME=/usr/local/nccl2 HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITHOUT_TENSORFLOW=1 HOROVOD_WITHOUT_PYTORCH=1 INCLUDES=['mxnet git directory'/python/mxnet/include] LIBRARY_DIRS=['mxnet git directory'/lib] /home/ubuntu/anaconda3/bin/pip install --upgrade -v --no-cache-dir ['horovod git directory']

An example script is: sudo PATH=/usr/local/bin:$PATH LD_LIBRARY_PATH=$LD_LIBRARY_PATH HOROVOD_NCCL_HOME=/usr/local/nccl2 HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITHOUT_TENSORFLOW=1 HOROVOD_WITHOUT_PYTORCH=1 INCLUDES=/home/ubuntu/master/python/mxnet/include LIBRARY_DIRS=/home/ubuntu/master/lib pip3 install --upgrade -v --no-cache-dir /home/ubuntu/horovod/

# Running
You can run the synthetic benchmark by doing (tested using OpenMPI 3.1.1 on AWS p3.16xlarge instances):

```mpirun -np 8 --hostfile ~/host_file --bind-to none --map-by slot -x NCCL_DEBUG=INFO -x NCCL_MIN_NRINGS=4 -x LD_LIBRARY_PATH -x PATH -x MXNET_USE_OPERATOR_TUNING=0 -mca pml ob1 -mca btl ^openib python3 mxnet_imagenet_resnet50.py --batch-size 128 --dtype float32 --data-nthreads 4 --num-epochs 90 --gpus 0 --lr 0.8 --warmup-epochs 10 --model resnet50_v1 --benchmark 1```

Note: the use of MXNET_USE_OPERATOR_TUNING=0 flag to disable OpenMP tuning. If this flag is not included, then starting up 8 MXNet processes will take upwards of 2 minutes. We find disabling this tuning does not affect performance.

To run on Imagenet data:

```mpirun -np 8 --hostfile ~/host_file --bind-to none --map-by slot -x NCCL_DEBUG=INFO -x NCCL_MIN_NRINGS=4 -x LD_LIBRARY_PATH -x PATH -x MXNET_USE_OPERATOR_TUNING=0 -mca pml ob1 -mca btl ^openib python3 mxnet_imagenet_resnet50.py --batch-size 128 --dtype float32 --data-nthreads 4 --num-epochs 90 --gpus 0 --lr 0.8 --warmup-epochs 10 --model resnet50_v1 --rec-train /media/ramdisk/train-passthrough.rec --rec-train-idx /media/ramdisk/train-passthrough.idx --rec-val /media/ramdisk/val-passthrough.rec --rec-val-idx /media/ramdisk/val-passthrough.idx```

# Testing
The following Horovod unit tests do not pass:
  * tests that check Horovod+MXNet throws the correct error if the user passes in NDArrays that differ in size
  
To run tests, we did:

```mpirun -np 8 --hostfile ~/host_file --bind-to none --map-by slot -x NCCL_DEBUG=INFO -x NCCL_MIN_NRINGS=4 -x LD_LIBRARY_PATH -x PATH -x MXNET_USE_OPERATOR_TUNING=0 -mca pml ob1 -mca btl ^openib python test_mxnet.py```

# Troubleshooting
Two common errors that happen are:
1. `OSError: libmxnet.so: cannot open shared object file: No such file or directory`

Solution: `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubuntu/master/lib`

2. `OSError: /home/ubuntu/horovod/horovod/mxnet/mpi_lib.cpython-36m-x86_64-linux-gnu.so: cannot open shared object file: No such file or directory`

Solution: launch Python from another directory

# Performance
Our scalability results are here with and without hierarchical allreduce (HA) on ResNet-50 with float32:
```
# gpus | Without HA |   With HA
---------------------------------
   8   |  3072  (NA)|  3078  (NA)
  16   |  6027 (98%)|  5859 (95%)
  32   | 12030 (98%)| 11675 (95%)
  64   | 22346 (83%)| 23166 (94%)
 128   | 40938 (84%)| 45972 (93%)
 256   | 64998 (66%)| 89858 (91%)
```

![Scalability](https://cwiki.apache.org/confluence/download/attachments/93323454/scalability.png?version=1&modificationDate=1538682773000&api=v2)
Figure 4. Preliminary benchmark on synthetic data comparing parameter server co-located (servers on same node as workers) and Horovod+MXNet

# References

(Design document)[https://cwiki.apache.org/confluence/display/MXNET/Horovod-MXNet+Integration]
