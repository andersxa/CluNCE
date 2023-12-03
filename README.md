# CluNCE
An Unsupervised Online Learning with Prototypical Cluster Streaming framework developed for the course 02830 Advanced Project in Digital Media Engineering at the Technical University of Denmark

# Installation:
Start by installing PyTorch from here: https://pytorch.org/get-started/locally/

And install Weights & Biases from here: https://docs.wandb.ai/quickstart

Then install this library as follows:
```
pip install pip install git+https://github.com/andersxa/clunce.git
```

# Usage:
``train.py`` is the main file for training the model. It takes the following arguments:
```
usage: train.py [-h] [--clustering CLUSTERING] [--batch-size N]
                [--test-batch-size N] [--epochs N] [--lr LR] [--seed S]
                [--log-interval N] [--eval-interval N] [--save-model]
                [--dataset DATASET] [--model-save-path MODEL_SAVE_PATH]
                [--data-path DATA_PATH] [--num-workers NUM_WORKERS]
                [--model MODEL] [--latent-dim LATENT_DIM]
                [--temperature T] [--num-update-batches NUM_UPDATE_BATCHES]
                [--max-iter MAX_ITER] [--queue-size QUEUE_SIZE]
                [--momentum MOMENTUM] [--warmup-epochs WARMUP_EPOCHS]
                [--num-prototypes NUM_PROTOTYPES] [--n-init N_INIT]
                [--num-initial-microclusters NUM_INITIAL_MICROCLUSTERS]
                [--delta-timestamp DELTA_TIMESTAMP] [--m-recent M_RECENT]
                [--max-boundary-factor MAX_BOUNDARY_FACTOR]
```

The ``--queue-size``, ``--momentum``, ``--warmup-epochs``, ``--num-prototypes`` and ``--n-init`` arguments are only used for ProtoNCE, while the ``--num-initial-microclusters``, ``--delta-timestamp``, ``--m-recent`` and ``--max-boundary-factor`` arguments are only used for CluNCE.

# Examples:
To train ProtoNCE on the ImageNet dataset with a ResNet50 model and a latent dimension of 128, a temperature of 0.2, a queue size of 8192 and 300 prototypes, run the following command:
```
python train.py --clustering protonce --dataset imagenet --model resnet50 --latent-dim 128 --temperature 0.2 --queue-size 8192 --num-prototypes 300
```
To train a CluNCE model on the ImageNet dataset with a ResNet50 model and a latent dimension of 128, a temperature of 0.2, a queue size of 8192 and 300 prototypes, run the following command:
```
python train.py --clustering clunce --dataset imagenet --model resnet50 --latent-dim 128 --temperature 0.2 --num-initial-microclusters 8192 --delta-timestamp 0.01
```
