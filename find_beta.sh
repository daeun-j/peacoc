cd data/utils
python3 run.py -d cifar10 -cn 10 -c 10 --num_val 1000
cd ../../

cd src/server

python3 peacoc_beta.py -d cifar10 -m res18 -ge 20 -le 20 -jr 1 -lr 0.1 --global_lr 1 -mom 0.9 -wd 0.0005 -bs 128 -vg 1 -T 1e-8 -fn beta_1e-8

