cat hosts | while read host;
do 
  echo $host
  scp download.sh ubuntu@$host:/home/ubuntu
  ssh -o "StrictHostKeyChecking no" $host 'sh download.sh' &
done
sh download.sh
wait
