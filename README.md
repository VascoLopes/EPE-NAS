# [EPE-NAS: Efficient Performance Estimation Without Training for Neural Architecture Search](https://arxiv.org/abs/2102.08099)

This repository contains code the paper, [EPE-NAS](https://arxiv.org/abs/2102.08099).

# Setup
## Datasets
1. Download the [datasets](https://drive.google.com/drive/folders/1L0Lzq8rWpZLPfiQGd6QR8q5xLV88emU7).

1.1 Place the folders in ~path_to_epenas/datasets/

## NAS-Bench-201
2. Download [NAS-Bench-201](https://drive.google.com/file/d/16Y0UwGisiouVRxW-W5hEtbxmcHw_0hF_/view) ([smaller version](https://drive.google.com/open?id=1SKW0Cu0u8-gb18zDpaAGi0f74UdXeGKs)).

2.1 Place the .pth file in ~path_to_epenas/datasets/

We also refer the reader to instructions in the official [NAS-Bench-201 README](https://github.com/D-X-Y/NAS-Bench-201).

## Requirements
3. Install the requirements in a conda environment with `conda env create -f environment.yml`.


# Reproducing our results
To reproduce our results:

```
conda activate epe-nas
./reproduce.sh 3 # average accuracy over 3 runs
```

Each command will finish by calling `process_results.py`, which will print a table. `./reproduce.sh 3` should print the following table:

| Method       |   Search time (s) | CIFAR-10 (val)   | CIFAR-10 (test)   | CIFAR-100 (val)   | CIFAR-100 (test)   | ImageNet16-120 (val)   | ImageNet16-120 (test)   | 
|:-------------|------------------:|:-----------------|:------------------|:------------------|:-------------------|:-----------------------|:------------------------|
| Ours (N=10)  |              2.77 | 89.90 +- 0.21    | 92.63 +- 0.32     | 69.78 +- 2.44     | 70.10 +- 1.71      | 41.73 +- 3.60          | 41.92 +- 4.25           |
| Ours (N=100) |             20.47 | 88.74 +- 3.16    | 91.59 +- 0.87     | 67.28 +- 3.68     | 67.19 +- 3.82      | 38.66 +- 4.75          | 38.80 +- 5.41           |
| Ours (N=500) |            105.84 | 88.17 +- 1.35    | 92.27 +- 1.75     | 69.23 +- 0.62     | 69.33 +- 0.66      | 41.93 +- 3.19          | 42.05 +- 3.09           |
| Ours (N=1000)|            206.23 | 87.87 +- 0.85    | 91.31 +- 1.69     | 69.44 +- 0.83     | 69.58 +- 0.83      | 41.86 +- 2.33          | 41.84 +- 2.06           |


The code is licensed under the MIT licence.

# Acknowledgements

This repository makes liberal use of code from the [AutoDL](https://github.com/D-X-Y/AutoDL-Projects) library, [NAS-Bench-201](https://github.com/D-X-Y/NAS-Bench-201) and [NAS-WOT](https://github.com/BayesWatch/nas-without-training). We are grateful to the authors for making the implementations publicly available.

# Citing us

If you use or build on our work, please consider citing us:

```bibtex
@misc{lopes2021epenas,
      title={EPE-NAS: Efficient Performance Estimation Without Training for Neural Architecture Search}, 
      author={Vasco Lopes and Saeid Alirezazadeh and Luís A. Alexandre},
      year={2021},
      eprint={2102.08099},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
