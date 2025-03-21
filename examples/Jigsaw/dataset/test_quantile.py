from transformers import pipeline, AutoTokenizer
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import plotly.express as px

tqdm.pandas()
# load the dataset
# jigsaw_df_train = pd.read_csv("jigsaw_train.csv")
# jigsaw_df_test = pd.read_csv("jigsaw_test.csv")
# imdb_df_train = pd.read_csv("imdb_train.csv")
# imdb_df_test = pd.read_csv("imdb_test.csv")

# imdb_df_test = imdb_df_test[imdb_df_test.index % 10 == 0]
# print("IMDB test size: ", len(imdb_df_test))


# tokenizer = AutoTokenizer.from_pretrained(
#     "/mnt/research/Anon2/Students/auth2/repos/trl/examples/Jigsaw/dataset/models_to_test/2024-01-18_01-23-52"
# )

# generation_kwargs = {
#     "min_length": 48,
#     "top_k": 50,
#     "top_p": 1.0,
#     "do_sample": True,
#     "pad_token_id": tokenizer.eos_token_id,
#     "max_new_tokens": 48,
# }

# generator = pipeline(
#     "text-generation",
#     model="/mnt/research/Anon2/Students/auth2/repos/trl/examples/Jigsaw/dataset/models_to_test/2024-01-18_01-23-52",
# )

# generator_risk = pipeline(
#     "text-generation",
#     model="/mnt/research/Anon2/Students/auth2/repos/trl/examples/Jigsaw/dataset/models_to_test/2024-01-18_09-45-09",
# )

# print("Generating completions for RLHF model")
# imdb_df_test["generation"] = imdb_df_test["text"].progress_apply(
#     lambda x: generator(x[:64], **generation_kwargs)[0]["generated_text"]
# )
# print("Generating completions for RLHF risk averse model")
# imdb_df_test["generation_risk"] = imdb_df_test["text"].progress_apply(
#     lambda x: generator_risk(x[:64], **generation_kwargs)[0]["generated_text"]
# )

# imdb_df_test.to_csv("imdb_generations.csv", index=False)

# # Reward computation
# sent_kwargs = {"return_all_scores": True, "function_to_apply": "none"}
# imdb_reward_model = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb", truncation=True, **sent_kwargs)


# print("Calculating rewards for RLHF model")
# imdb_df_test["generation_reward"] = imdb_df_test["generation"].progress_apply(
#     lambda x: imdb_reward_model(x)[0][1]["score"]
# )

# print("Calculating rewards for RLHF risk averse model")
# imdb_df_test["generation_risk_reward"] = imdb_df_test["generation_risk"].progress_apply(
#     lambda x: imdb_reward_model(x)[0][1]["score"]
# )

# imdb_df_test.to_csv("imdb_generations.csv", index=False)

# Plot reward distribution for RLHF and RLHF risk averse model

# Load final dataset
imdb_df_test = pd.read_csv("jigsaw_generations.csv")
imdb_df_test = imdb_df_test.rename(columns={"R": "RLHF", "R_risk": "RLHF risk averse"})
fig = px.ecdf(
    imdb_df_test,
    x=["RLHF", "RLHF risk averse"],
    labels={
        "value": "Reward",
        "variable": "Model",
    },
)
fig.update_layout(
    title="Reward distribution for RLHF and RLHF risk averse models",
    xaxis_title="Reward",
    yaxis_title="CDF",
)
print("Average reward for RLHF: ", imdb_df_test["RLHF"].mean())
print("Average reward for RLHF risk averse: ", imdb_df_test["RLHF risk averse"].mean())
fig.write_image("jigsaw_R_cdf.png")

imdb_df_test = imdb_df_test.rename(columns={"R_bar": "RLHF returns", "R_bar_risk": "RLHF risk averse returns"})
fig = px.ecdf(
    imdb_df_test,
    x=["RLHF returns", "RLHF risk averse returns"],
    labels={
        "value": "Reward",
        "variable": "Model",
    },
)
fig.update_layout(
    title="Returns distribution for RLHF and RLHF risk averse models",
    xaxis_title="Returns",
    yaxis_title="CDF",
)
print("Average return for RLHF: ", imdb_df_test["RLHF returns"].mean())
print("Average return for RLHF risk averse: ", imdb_df_test["RLHF risk averse returns"].mean())
fig.write_image("jigsaw_R_bar_cdf.png")
