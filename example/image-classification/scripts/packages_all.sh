cat hosts | while read host;
do 
  echo $host
  scp packages.sh ubuntu@$host:/home/ubuntu
  scp /home/ubuntu/nccl-repo-ubuntu1604-2.1.15-ga-cuda9.0_1-1_amd64.deb ubuntu@$host:/home/ubuntu
  ssh -o "StrictHostKeyChecking no" $host 'sh packages.sh' &
done
sh packages.sh
wait
