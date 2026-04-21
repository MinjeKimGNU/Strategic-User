# 212 121 21 2
for seed in 2120
do
    python run_recbole_seed.py --model=SASRec --seed=$seed --dataset=Amazon_Beauty --config_files='configs/SASRec/Amazon_Beauty.yaml'
done


for seed in 2120
do
    python run_recbole_seed.py --model=SASRec --seed=$seed --dataset=Amazon_Sports_and_Outdoors --config_files='configs/SASRec/Amazon_Sports_and_Outdoors.yaml'
done


for seed in 2120
do
    python run_recbole_seed.py --model=SASRec --seed=$seed --dataset=Amazon_Toys_and_Games --config_files='configs/SASRec/Amazon_Toys_and_Games.yaml'
done

for seed in 2120
do
    python run_recbole_seed.py --model=SASRec --seed=$seed --dataset=yelp --config_files='configs/SASRec/yelp.yaml'
done

"""


for seed in 2120 212 121 21 2
do
    python run_recbole_seed.py --model=SASRecF --seed=$seed --dataset=Amazon_Beauty --config_files='configs/SASRecF/Amazon_Beauty.yaml'
done


for seed in 2120 212 121 21 2
do
    python run_recbole_seed.py --model=SASRecF --seed=$seed --dataset=Amazon_Sports_and_Outdoors --config_files='configs/SASRecF/Amazon_Sports_and_Outdoors.yaml'
done


for seed in 2120 212 121 21 2
do
    python run_recbole_seed.py --model=SASRecF --seed=$seed --dataset=Amazon_Toys_and_Games --config_files='configs/SASRecF/Amazon_Toys_and_Games.yaml'
done

for seed in 2120 212 121 21 2
do
    python run_recbole_seed.py --model=SASRecF --seed=$seed --dataset=yelp --config_files='configs/SASRecF/yelp.yaml'
done


"""