cd data/utils
python3 run.py -d cifar10 -cn 10 -c 10 --num_val 1000
cd ../../

cd src/server
# don't change ge, le 
python3 fedavg.py  -d cifar10 -m res18 -ge 20 -le 20 -jr 1 -lr 0.01 -mom 0.9 -wd 0.0005 -bs 512 -vg 1 >  ../avg.out
python3 fedprox.py -d cifar10 -m res18 -ge 20 -le 20 -jr 1 -lr 0.01 -mom 0.9 -wd 0.0005 -bs 512 -vg 1 > ../prox.out
python3 peacoc.py -d cifar10 -m res18 -ge 20 -le 20 -jr 1 -lr 0.01 -mom 0.9 -wd 0.0005 -bs 512 -vg 1 > ../pcoc.out
python3 peacoc_beta.py -d cifar10 -m res18 -ge 20 -le 20 -jr 1 -lr 0.01 -mom 0.9 -wd 0.0005 -bs 512  -vg 1 > ../pcocbeta.out
