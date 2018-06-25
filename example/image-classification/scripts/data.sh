mkdir -p ~/data/
aws s3 sync s3://sagemaker-hpo/data/imagenet/pass-through/ ~/data/
sudo mkdir -p /media/ramdisk
sudo mount -t tmpfs -o size=200G tmpfs /media/ramdisk
cp -r ~/data/pass-through/ /media/ramdisk/
