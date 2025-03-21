## **Risk Averse RLHF for fine tuning LLMs**

This code is based on the huggngface trl github repository  [trl](https://github.com/huggingface/trl)

Follow the setup instructions as mentioned in the trl repository

To run the experiments, execute the following commands, 

**IMDB training :**
```
cd examples/IMDB/training

sh ppo_run_single_script.sh #RLHF
sh sr_ppo_run_single_script.sh #RA-RLHF
```
**Jigsaw training :**
```
cd examples/Jigsaw/training

sh ppo_run_single_script.sh #RLHF
sh sr_ppo_run_single_script.sh #RA-RLHF
```
**GPT-J 6B IMDB training:**
```
cd examples/IMDB/training

python ppo_big_imdb.py #RLHF
python sr_ppo_big_imdb.py #RA-RLHF
```