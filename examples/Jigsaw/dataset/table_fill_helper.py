import pandas as pd
import matplotlib.pyplot as plt

# import plotly.express as px

import pandas as pd

import numpy as np

import pdb

import statistics as st


# Jigsaw
jigsaw_gpt = pd.read_csv("jigsaw_generations_8_32_alpha_40_seed_2.csv")[["R_gpt2", "perplexity_gpt2"]].copy()
jigsaw_1 = pd.read_csv("jigsaw_generations_8_32_alpha_20_seed_2.csv")
jigsaw_1 = pd.concat([jigsaw_1, jigsaw_gpt], axis=1)
# select all sft columns and put into new dataframe
jigsaw_sft = jigsaw_1[["R_sft", "perplexity_sft"]].copy()

jigsaw_2 = pd.read_csv("jigsaw_generations_8_32_alpha_20_seed_94.csv")
jigsaw_2 = pd.concat([jigsaw_2, jigsaw_sft, jigsaw_gpt], axis=1)
jigsaw_3 = pd.read_csv("jigsaw_generations_8_32_alpha_20_seed_56.csv")
jigsaw_3 = pd.concat([jigsaw_3, jigsaw_sft, jigsaw_gpt], axis=1)


print(jigsaw_1.columns)
# a = input()

# pdb.set_trace()


# average env reward

sft_means = [jigsaw_1["R_sft"].mean(), jigsaw_2["R_sft"].mean(), jigsaw_3["R_sft"].mean()]
# print(sft_means)
gpt2_means = [jigsaw_1["R_gpt2"].mean(), jigsaw_2["R_gpt2"].mean(), jigsaw_3["R_gpt2"].mean()]
rlhf_means = [jigsaw_1["R"].mean(), jigsaw_2["R"].mean(), jigsaw_3["R"].mean()]
ra_rlhf_means = [jigsaw_1["R_risk"].mean(), jigsaw_2["R_risk"].mean(), jigsaw_3["R_risk"].mean()]


# print average env reward mean and var

print("gpt2 r mean and var:", round(st.mean(gpt2_means), 3), round(st.stdev(gpt2_means), 3))
print("sft r mean and var:", round(st.mean(sft_means), 3), round(st.stdev(sft_means), 3))
print("rlhf r mean and var:", round(st.mean(rlhf_means), 3), round(st.stdev(rlhf_means), 3))
print("ra rlhf r mean and var:", round(st.mean(ra_rlhf_means), 3), round(st.stdev(ra_rlhf_means), 3))

# perplexities

# gpt2_ppl = [jigsaw_1["perplexity_gpt2"].mean(), jigsaw_2["perplexity_gpt2"].mean(), jigsaw_3["perplexity_gpt2"].mean()]
# sft_ppl = [jigsaw_1["perplexity_sft"].mean(), jigsaw_2["perplexity_sft"].mean(), jigsaw_3["perplexity_sft"].mean()]
# rlhf_ppl = [jigsaw_1["perplexity"].mean(), jigsaw_2["perplexity"].mean(), jigsaw_3["perplexity"].mean()]
# ra_rlhf_ppl = [
#     jigsaw_1["perplexity_risk"].mean(),
#     jigsaw_2["perplexity_risk"].mean(),
#     jigsaw_3["perplexity_risk"].mean(),
# ]

# # print ppl mean and var

# print(f"ra_rlhf_ppl: {ra_rlhf_ppl}", f"rlhf+ppl{rlhf_ppl}")

# print("gpt2 ppl mean and var:", round(st.mean(gpt2_ppl), 3), round(st.stdev(gpt2_ppl), 3))
# print("sft ppl mean and var:", round(st.mean(sft_ppl), 3), round(st.stdev(sft_ppl), 3))
# print("rlhf ppl mean and var:", round(st.mean(rlhf_ppl), 3), round(st.stdev(rlhf_ppl), 3))
# print("ra rlhf ppl mean and var:", round(st.mean(ra_rlhf_ppl), 3), round(st.stdev(ra_rlhf_ppl), 3))


# tail performance

tail_to_save = 5
print(f"\n\ntail_to_save: {tail_to_save}")

tail_1 = jigsaw_1[jigsaw_1["prompt_score"] <= tail_to_save]

tail_2 = jigsaw_2[jigsaw_2["prompt_score"] <= tail_to_save]

tail_3 = jigsaw_3[jigsaw_3["prompt_score"] <= tail_to_save]


gpt2_tl_means = [tail_1["R_gpt2"].mean(), tail_2["R_gpt2"].mean(), tail_3["R_gpt2"].mean()]
sft_tl_means = [tail_1["R_sft"].mean(), tail_2["R_sft"].mean(), tail_3["R_sft"].mean()]
rlhf_tl_means = [tail_1["R"].mean(), tail_2["R"].mean(), tail_3["R"].mean()]
ra_rlhf_tl_means = [tail_1["R_risk"].mean(), tail_2["R_risk"].mean(), tail_3["R_risk"].mean()]

print("gpt2 tail mean and var:", round(st.mean(gpt2_tl_means), 3), round(st.stdev(gpt2_tl_means), 3))
print("sft tail mean and var:", round(st.mean(sft_tl_means), 3), round(st.stdev(sft_tl_means), 3))
print("rlhf tail mean and var:", round(st.mean(rlhf_tl_means), 3), round(st.stdev(rlhf_tl_means), 3))
print("ra rlhf tail mean and var:", round(st.mean(ra_rlhf_tl_means), 3), round(st.stdev(ra_rlhf_tl_means), 3))

tail_to_save = 2.5
print(f"\n\ntail_to_save: {tail_to_save}")

tail_1 = jigsaw_1[jigsaw_1["prompt_score"] <= tail_to_save]

tail_2 = jigsaw_2[jigsaw_2["prompt_score"] <= tail_to_save]

tail_3 = jigsaw_3[jigsaw_3["prompt_score"] <= tail_to_save]


gpt2_tl_means = [tail_1["R_gpt2"].mean(), tail_2["R_gpt2"].mean(), tail_3["R_gpt2"].mean()]
sft_tl_means = [tail_1["R_sft"].mean(), tail_2["R_sft"].mean(), tail_3["R_sft"].mean()]
rlhf_tl_means = [tail_1["R"].mean(), tail_2["R"].mean(), tail_3["R"].mean()]
ra_rlhf_tl_means = [tail_1["R_risk"].mean(), tail_2["R_risk"].mean(), tail_3["R_risk"].mean()]

print("gpt2 tail mean and var:", round(st.mean(gpt2_tl_means), 3), round(st.stdev(gpt2_tl_means), 3))
print("sft tail mean and var:", round(st.mean(sft_tl_means), 3), round(st.stdev(sft_tl_means), 3))
print("rlhf tail mean and var:", round(st.mean(rlhf_tl_means), 3), round(st.stdev(rlhf_tl_means), 3))
print("ra rlhf tail mean and var:", round(st.mean(ra_rlhf_tl_means), 3), round(st.stdev(ra_rlhf_tl_means), 3))

tail_to_save = 0.0
print(f"\n\ntail_to_save: {tail_to_save}")

tail_1 = jigsaw_1[jigsaw_1["prompt_score"] <= tail_to_save]

tail_2 = jigsaw_2[jigsaw_2["prompt_score"] <= tail_to_save]

tail_3 = jigsaw_3[jigsaw_3["prompt_score"] <= tail_to_save]


gpt2_tl_means = [tail_1["R_gpt2"].mean(), tail_2["R_gpt2"].mean(), tail_3["R_gpt2"].mean()]
sft_tl_means = [tail_1["R_sft"].mean(), tail_2["R_sft"].mean(), tail_3["R_sft"].mean()]
rlhf_tl_means = [tail_1["R"].mean(), tail_2["R"].mean(), tail_3["R"].mean()]
ra_rlhf_tl_means = [tail_1["R_risk"].mean(), tail_2["R_risk"].mean(), tail_3["R_risk"].mean()]

print("gpt2 tail mean and var:", round(st.mean(gpt2_tl_means), 3), round(st.stdev(gpt2_tl_means), 3))
print("sft tail mean and var:", round(st.mean(sft_tl_means), 3), round(st.stdev(sft_tl_means), 3))
print("rlhf tail mean and var:", round(st.mean(rlhf_tl_means), 3), round(st.stdev(rlhf_tl_means), 3))
print("ra rlhf tail mean and var:", round(st.mean(ra_rlhf_tl_means), 3), round(st.stdev(ra_rlhf_tl_means), 3))

tail_to_save = -2.5
print(f"\n\ntail_to_save: {tail_to_save}")

tail_1 = jigsaw_1[jigsaw_1["prompt_score"] <= tail_to_save]

tail_2 = jigsaw_2[jigsaw_2["prompt_score"] <= tail_to_save]

tail_3 = jigsaw_3[jigsaw_3["prompt_score"] <= tail_to_save]


gpt2_tl_means = [tail_1["R_gpt2"].mean(), tail_2["R_gpt2"].mean(), tail_3["R_gpt2"].mean()]
sft_tl_means = [tail_1["R_sft"].mean(), tail_2["R_sft"].mean(), tail_3["R_sft"].mean()]
rlhf_tl_means = [tail_1["R"].mean(), tail_2["R"].mean(), tail_3["R"].mean()]
ra_rlhf_tl_means = [tail_1["R_risk"].mean(), tail_2["R_risk"].mean(), tail_3["R_risk"].mean()]

print("gpt2 tail mean and var:", round(st.mean(gpt2_tl_means), 3), round(st.stdev(gpt2_tl_means), 3))
print("sft tail mean and var:", round(st.mean(sft_tl_means), 3), round(st.stdev(sft_tl_means), 3))
print("rlhf tail mean and var:", round(st.mean(rlhf_tl_means), 3), round(st.stdev(rlhf_tl_means), 3))
print("ra rlhf tail mean and var:", round(st.mean(ra_rlhf_tl_means), 3), round(st.stdev(ra_rlhf_tl_means), 3))

tail_to_save = -5
print(f"\n\ntail_to_save: {tail_to_save}")

tail_1 = jigsaw_1[jigsaw_1["prompt_score"] <= tail_to_save]

tail_2 = jigsaw_2[jigsaw_2["prompt_score"] <= tail_to_save]

tail_3 = jigsaw_3[jigsaw_3["prompt_score"] <= tail_to_save]


gpt2_tl_means = [tail_1["R_gpt2"].mean(), tail_2["R_gpt2"].mean(), tail_3["R_gpt2"].mean()]
sft_tl_means = [tail_1["R_sft"].mean(), tail_2["R_sft"].mean(), tail_3["R_sft"].mean()]
rlhf_tl_means = [tail_1["R"].mean(), tail_2["R"].mean(), tail_3["R"].mean()]
ra_rlhf_tl_means = [tail_1["R_risk"].mean(), tail_2["R_risk"].mean(), tail_3["R_risk"].mean()]

print("gpt2 tail mean and var:", round(st.mean(gpt2_tl_means), 3), round(st.stdev(gpt2_tl_means), 3))
print("sft tail mean and var:", round(st.mean(sft_tl_means), 3), round(st.stdev(sft_tl_means), 3))
print("rlhf tail mean and var:", round(st.mean(rlhf_tl_means), 3), round(st.stdev(rlhf_tl_means), 3))
print("ra rlhf tail mean and var:", round(st.mean(ra_rlhf_tl_means), 3), round(st.stdev(ra_rlhf_tl_means), 3))
