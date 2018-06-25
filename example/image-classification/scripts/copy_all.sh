cat hosts | while read host;
do 
  echo $host
  scp copy.sh ubuntu@$host:/home/ubuntu
  ssh -o "StrictHostKeyChecking no" $host 'sh copy.sh' &
done
sh copy.sh
wait
