#!/bin/bash

python search.py --dataset cifar10 --data_loc './datasets/cifar10'            --n_runs $1 --n_samples 10 --api_loc './datasets/NAS-Bench-201-v1_0-e61699.pth'
python search.py --dataset cifar10 --trainval --data_loc './datasets/cifar10' --n_runs $1 --n_samples 10 --api_loc './datasets/NAS-Bench-201-v1_0-e61699.pth'
python search.py --dataset cifar100 --data_loc './datasets/cifar100'          --n_runs $1 --n_samples 10 --api_loc './datasets/NAS-Bench-201-v1_0-e61699.pth'
python search.py --dataset ImageNet16-120 --data_loc './datasets/ImageNet16'  --n_runs $1 --n_samples 10 --api_loc './datasets/NAS-Bench-201-v1_0-e61699.pth'

python search.py --dataset cifar10 --data_loc './datasets/cifar10'            --n_runs $1 --n_samples 100 --api_loc './datasets/NAS-Bench-201-v1_0-e61699.pth'
python search.py --dataset cifar10 --trainval --data_loc './datasets/cifar10' --n_runs $1 --n_samples 100 --api_loc './datasets/NAS-Bench-201-v1_0-e61699.pth'
python search.py --dataset cifar100 --data_loc './datasets/cifar100'          --n_runs $1 --n_samples 100 --api_loc './datasets/NAS-Bench-201-v1_0-e61699.pth'
python search.py --dataset ImageNet16-120 --data_loc './datasets/ImageNet16'  --n_runs $1 --n_samples 100 --api_loc './datasets/NAS-Bench-201-v1_0-e61699.pth'

python search.py --dataset cifar10 --data_loc './datasets/cifar10'            --n_runs $1 --n_samples 500 --api_loc './datasets/NAS-Bench-201-v1_0-e61699.pth'
python search.py --dataset cifar10 --trainval --data_loc './datasets/cifar10' --n_runs $1 --n_samples 500 --api_loc './datasets/NAS-Bench-201-v1_0-e61699.pth'
python search.py --dataset cifar100 --data_loc './datasets/cifar100'          --n_runs $1 --n_samples 500 --api_loc './datasets/NAS-Bench-201-v1_0-e61699.pth'
python search.py --dataset ImageNet16-120 --data_loc './datasets/ImageNet16'  --n_runs $1 --n_samples 500 --api_loc './datasets/NAS-Bench-201-v1_0-e61699.pth'

python search.py --dataset cifar10 --data_loc './datasets/cifar10'            --n_runs $1 --n_samples 1000 --api_loc './datasets/NAS-Bench-201-v1_0-e61699.pth'
python search.py --dataset cifar10 --trainval --data_loc './datasets/cifar10' --n_runs $1 --n_samples 1000 --api_loc './datasets/NAS-Bench-201-v1_0-e61699.pth'
python search.py --dataset cifar100 --data_loc './datasets/cifar100'          --n_runs $1 --n_samples 1000 --api_loc './datasets/NAS-Bench-201-v1_0-e61699.pth'
python search.py --dataset ImageNet16-120 --data_loc './datasets/ImageNet16'  --n_runs $1 --n_samples 1000 --api_loc './datasets/NAS-Bench-201-v1_0-e61699.pth'

python process_results.py --n_runs $1