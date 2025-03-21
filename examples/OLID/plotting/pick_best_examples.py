import pandas as pd

df = pd.read_csv("../dataset/jigsaw_generations_8_32_alpha_20_seed_2.csv")
df = df.filter(["query", "prompt_score", "response", "R", "response_risk", "R_risk"])

# Sort by difference between R and R_risk
df["R_diff"] = df["R_risk"] - df["R"]
df = df.sort_values(by=["R_diff"], ascending=False)

# remove columns where prompt score bigger than R risk or R
df = df[df["prompt_score"] < df["R_risk"]]
df = df[df["prompt_score"] < df["R"]]
df = df[df["prompt_score"] < -2.5]

df.to_csv("examples.csv", index=False)
