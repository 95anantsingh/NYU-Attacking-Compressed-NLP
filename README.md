<div align="center">

# <b>Attacking Compressed NLP</b>

[![LICENSE](https://img.shields.io/badge/License-GPL%203.0-blue.svg)](https://github.com/95anantsingh/NYU-Attacking-Compressed-NLP/blob/master/LICENSE) [![PYTHON](https://img.shields.io/badge/python-v3.8-yellow.svg)]() [![PYTORCH](https://img.shields.io/badge/PyTorch-v1.11-red.svg)](https://pytorch.org/) [![CUDA](https://img.shields.io/badge/CUDA-v11.3-green.svg)](https://developer.nvidia.com/cuda-11.3.0-download-archive) 

</div>

This project aims to investigate the transferability of adversarial samples across the sate of the art NLP models and their compressed versions and infer the effects different compression techniques have on adversarial attacks.

<br>

## :wrench: Dependencies and Installation

### Dependencies
- [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Python = 3.8
- [PyTorch = 1.11](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads) = 11.3
- Linux


### Installation

1. Clone repo

    ```bash
    git clone https://github.com/95anantsingh/NYU-Attacking-Compressed-NLP.git
    cd NYU-Attacking-Compressed-NLP
    ```

1. Create conda environment

    ```bash
    conda env create -f environment.yml
    ```

1. Download BERT model weights

    ```bash
    wget -i bert_weight_urls --directory-prefix models/data/weights
    ```

1. Download LSTM model weights

    ```bash
    wget -i lstm_weight_urls --directory-prefix models/data/weights
    ```


Additionally the big pretrained models are stored on a drive link: please download them and store them to the corresponding location, more details in individual READMEs.
<br>


## :file_folder: Project Structure

This repo is structured:

-> BERT based SST attacks folder: see documentation [here](bert/sst/README.md)

-> LSTM based SST attacks folder: see documentation [here](lstm/sst/README.md)

<br>

## :books: Datasets

Dataset used: https://huggingface.co/datasets/sst, will be downloaded automatically on running the code

<br>

## :zap: Quick Inference

The instruction to run the code and description of the files in each folder is in a separate `README.md` inside the folder.

```bash
conda activate NLPattack
cd models/bert/sst
```

<br>

## :blue_book: Documentation

Project presentation and results can be found at [docs/presentation.pdf](https://github.com/95anantsingh/NYU-Attacking-Compressed-NLP/blob/master/docs/presentation.pdf)
<br>
Demo video can be downloaded from [docs/attack-demo.webm](https://github.com/95anantsingh/NYU-Attacking-Compressed-NLP/blob/master/docs/attack-demo.webm)

<br>

## :scroll: License

This repo is licensed under GPL License Version 3.0

<br>

## :e-mail: Contact

If you have any question, please email `anant.singh@nyu.edu` or `sp6646@nyu.edu`
<br> 
> This Project was part of graduate level High Performance Machine Learning course at New York University
