## ATTACKING COMPRESSED NLP

Investigate the transferability of adversarial samples across the SOTA NLP models and their compressed versions and infer the effects different compression techniques have on adversarial attacks


Dataset used: https://huggingface.co/datasets/sst, will be downloaded automatically on running the code

How this repo is structured:

-> BERT based SST attacks folder: see documentation [here](bert/sst/README.md)

-> LSTM based SST attacks folder: see documentation [here](lstm/sst/README.md)

Please use the `environment.yml` with conda to get the environment setup correctly

Additionally the big pretrained models are stored on a drive link: please download them and store them to the corresponding location, more details in individual READMEs.

The instruction to run the code and description of the files in each folder is in a separate `README.md` inside the folder.

Results: can be found in our slide deck [here](hpml presentation_new.pdf)
