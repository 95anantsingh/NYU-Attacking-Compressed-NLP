## File description:

Download following files to the data folder:
```
! cd data

! wget https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/train.txt

! wget https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/dev.txt

! wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
```
-Key Files

`run_create_uat_and_eval_attacks.ipynb`- used to run the `create_sst_uat.py` file and the`eval_triggers.py` file for various configurations to create white box triggers and perform transfer of the attacks

`quant_attacks.ipynb` - perform Black box attacks on the and transfer them

`create_sst_uat2.py` - used to create the universal adversarial attacks

`eval_triggers.py`- used to transfer the universal adversarial attacks between the different models

The pretrained models are stored on drive: `https://drive.google.com/drive/folders/1oEwxZ-nZF8JZFAWJYrkfQT3h4_7jqbBx?usp=sharing`
please download them to the respective folders.

-To train the models (you shouldnt need to do this):

`train_lstm.py` - used to train the main lstm model

`distillation.ipynb` - used to self-distil the LSTM model

-Finally, the output files are stored in this format:

`uat_<attacked class>_<targeted class>.txt` - stores the universal triggers

`eval_uat_<attacker model>_<attacked model>_<targeted class>.txt` - stores the transfer experiments results

`quantized_textfooler_attacks.csv` - stores the Black box attacks created on the quantized model

`main_textfooler_attacks.csv` - stores the Black box attacks created on the main model
