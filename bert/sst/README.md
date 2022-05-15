File description:

`run_create_uat_sst.ipynb` - used to run the `create_sst_uat.py` file for various configurations and create white box attacks
`run_eval_attacks.ipynb`- used to run the `eval_triggers.py` file for various configurations to perform transfer of the attacks
`create_sst_uat.py` - used to create the universal adversarial attacks
`eval_triggers.py`- used to transfer the universal adversarial attacks between the different models

To train the models(you shouldnt need to do this, the pretrained models are stored):
`train_bert.py` - used to train the main model
`training_distilled_bert.ipynb` - used to do knowledge distillation for the small finetuned bert model
`training_distilbert.ipynb` - used to finetune the distilbert model
`training_bert_pruned.ipynb` - used to prune the bert model

Finally, the output files are stored in this format:
`uat_<attacked class>_<targeted class>.txt` - stores the universal triggers
`eval_uat_<attacker model>_<attacked model>_<targeted class>.txt` - stores the transfer experiments results
