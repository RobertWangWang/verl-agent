train_batch_size=16
val_batch_size=128
group_size=5

source ~/miniconda3/etc/profile.d/conda.sh

# start appworld server
conda activate appworld
pkill -f "appworld serve environment"
sleep 10

total_instances=$((train_batch_size * group_size + val_batch_size))
for ((i=0; i<total_instances; i++))
do
    port=$((8000 + i))
    echo "Starting app on port $port"
    appworld serve environment --port $port &
done

sleep 10
