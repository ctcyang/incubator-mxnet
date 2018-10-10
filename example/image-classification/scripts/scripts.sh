#Synthetic data:

mpirun -np 8 --hostfile ~/host_file --bind-to none --map-by slot -x NCCL_DEBUG=INFO -x NCCL_MIN_NRINGS=4 -x LD_LIBRARY_PATH -x PATH -x MXNET_USE_OPERATOR_TUNING=0 -mca pml ob1 -mca btl ^openib python3 mxnet_imagenet_resnet50.py --batch-size 128 --dtype float32 --data-nthreads 4 --num-epochs 90 --gpus 0 --lr 0.8 --warmup-epochs 10 --model resnet50_v1 --benchmark 1

#Real data:

mpirun -np 8 --hostfile ~/host_file --bind-to none --map-by slot -x NCCL_DEBUG=INFO -x NCCL_MIN_NRINGS=4 -x LD_LIBRARY_PATH -x PATH -x MXNET_USE_OPERATOR_TUNING=0 -mca pml ob1 -mca btl ^openib python3 mxnet_imagenet_resnet50.py --batch-size 128 --dtype float32 --data-nthreads 4 --num-epochs 90 --gpus 0 --lr 0.8 --warmup-epochs 10 --model resnet50_v1 --rec-train /media/ramdisk/train-passthrough.rec --rec-train-idx /media/ramdisk/train-passthrough.idx --rec-val /media/ramdisk/val-passthrough.rec --rec-val-idx /media/ramdisk/val-passthrough.idx

#Unit tests:

mpirun -np 8 --hostfile ~/host_file --bind-to none --map-by slot -x NCCL_DEBUG=INFO -x NCCL_MIN_NRINGS=4 -x LD_LIBRARY_PATH -x PATH -x MXNET_USE_OPERATOR_TUNING=0 -mca pml ob1 -mca btl ^openib python test_mxnet.py

#Hierarchical allreduce on synthetic data:

mpirun -np 8 --hostfile ~/host_file --bind-to none --map-by slot -x HOROVOD_HIERARCHICAL_ALLREDUCE=1 -x NCCL_DEBUG=INFO -x NCCL_MIN_NRINGS=4 -x LD_LIBRARY_PATH -x PATH -x MXNET_USE_OPERATOR_TUNING=0 -mca pml ob1 -mca btl ^openib python3 mxnet_imagenet_resnet50.py --batch-size 128 --dtype float32 --data-nthreads 4 --num-epochs 90 --gpus 0 --lr 0.8 --warmup-epochs 10 --model resnet50_v1 --benchmark 1