cd data/utils
python3 run.py -d cifar10 -cn 10 -c 10
cd ../../

# cd src/server
# python3 peacoc.py -d cifar10 -m res18 -ge 3 -le 10 -jr 1 -lr 0.1 -mom 0.9 -wd 0.0005 -bs 128 -vg 1 -ab B > ../test.out
