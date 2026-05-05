# Logit Distance Bounds Representational Similarity
Code for the article: ["Logit Distance Bounds Representational Similarity"](https://arxiv.org/abs/2602.15438) by Beatrix M. G. Nielsen, Emanuele Marconato, Luigi Gresele, Andrea Dittadi and Simon Buchholz.

[![DOI](https://zenodo.org/badge/1066371626.svg)](https://doi.org/10.5281/zenodo.18667837)




## Description
Training models and calculating metrics can be done using main.py. See the "Usage" section  below for details.
Code for the plots for the examples in appendices G and H are in the "examples" folder.


## Installation guide

Use the environment.yaml file to install the required packages. 
```
conda env create -f environment.yml
```
Note that this expects you to be on a GPU machine. If you are on a CPU only machine you need a different setup for pytorch. 



## Datasets
We use three datasets. 

1. A synthetic dataset (dsets/synthetic.py)

2. [CIFAR-100](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.CIFAR100.html) loaded with the torchvision package. Originally from \[1\].

3. A bird image dataset called SUB from \[2\], a synthetic variant of CUB200 \[3\] containing bird images from 33 classes

References:

\[1\] Krizhevsky, A., Hinton, G., et al. Learning multiple layers of features from tiny images. 2009.

\[2\] Bader, J., Girrbach, L., Alaniz, S., and Akata, Z. Sub: Benchmarking cbm generalization via synthetic attribute substitutions. Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025.

\[3\] Wah, C., Branson, S., Welinder, P., Perona, P., and Belongie, S. The caltech-ucsd birds-200-2011 dataset. 2011.


## Usage

This code can train teacher models on one of the three datasets and distill student models using one of three distillation objectives:  
1. KL divergence to the teacher
2. L1 based logit loss 
3. L2 based logit loss. 

Examples for training teacher and student models on the three datasets are in the \[dataset\]_teacher_example and \[dataset\]_student_example files. 
The cifar100 files give a slurm job example. All possible arguments can be found in utils/args.py. 






## Acknowledgments

We thank Anton Rask Lundborg for pointers to the compositional data analysis literature and David Klindt for interesting discussions. 
B. M. G. N. was supported by the Danish Pioneer Centre for AI, DNRF grant number P1 and partially by the Novo Nordisk Foundation grant NNF24OC0092612.
E.M. acknowledges support from TANGO, Grant Agreement No. 101120763, funded by the European Union. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Health and Digital Executive Agency (HaDEA). Neither the European Union nor the granting authority can be held responsible for them. S.B.\ was supported by the Tübingen AI Center.
L.G.\ was supported by the Danish Data Science Academy, which is funded by the Novo Nordisk Foundation (NNF21SA0069429), and by the Pioneer Centre for AI, DNRF grant number P1.

## License 

See LICENSE.




