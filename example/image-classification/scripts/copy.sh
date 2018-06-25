sudo rm -rf /media/ramdisk/*
sudo mkdir -p /media/ramdisk/pass-through/
sudo mount -t tmpfs -o size=200G tmpfs /media/ramdisk/pass-through/
cp -r ~/data/* /media/ramdisk/pass-through/
