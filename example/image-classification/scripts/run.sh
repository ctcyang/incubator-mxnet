while read -u 10 host; do scp -o "StrictHostKeyChecking no" data.sh $host: ; done 10<24-hosts
while read -u 10 host; do ssh -o "StrictHostKeyChecking no" $host "tmux new-session -d bash data.sh" ; done 10<24-hosts
while read -u 10 host; do scp -o "StrictHostKeyChecking no" packages.sh $host: ; done 10<24-hosts
while read -u 10 host; do ssh -o "StrictHostKeyChecking no" $host "tmux new-session -d bash packages.sh" ; done 10<24-hosts
