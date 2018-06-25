cat ~/efs/8p-16r-hosts | while read host;
do 
  echo $host
  scp download.sh ubuntu@$host:/home/ubuntu
  ssh -o "StrictHostKeyChecking no" $host 'pkill -9 python' &
done
pkill -9 python
wait
