## **Code for Neurips 2024 paper 'Risk Averse fine tuning of LLMs'**

Here is the [Openreview](https://openreview.net/forum?id=1BZKqZphsW) link for the paper. 

This code is based on the huggngface trl github repository  [trl](https://github.com/huggingface/trl). Trained model checkpoints will be added shortly. 

Please follow the setup instructions as mentioned in the trl repository.

To run the experiments, execute the following commands, 

**IMDB training :**
```
git checkout  auth1/main

cd examples/IMDB/training

sh ppo_run_single_script.sh #RLHF
sh sr_ppo_run_single_script.sh #RA-RLHF
```
**Jigsaw training :**
```
git checkout  auth2/main

cd examples/Jigsaw/training

sh ppo_run_single_script.sh #RLHF
sh sr_ppo_run_single_script.sh #RA-RLHF
```
**GPT-J 6B IMDB training:**
```
git checkout  auth2/main

cd examples/IMDB/training

python ppo_big_imdb.py #RLHF
python sr_ppo_big_imdb.py #RA-RLHF
```