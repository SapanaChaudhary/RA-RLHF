import matplotlib.pyplot as plt
import pandas as pd


# read imdb_gen_df
imdb_gen_df = pd.read_csv(
    "/mnt/research/Anon2/Students/auth2/repos/trl/examples/OLID/dataset/imdb_generations_64_48_alpha_40_seed_4_full.csv"
)
# imdb_gen_df2 = pd.read_csv("imdb_generations_seed_42.csv")

# Replace columns of imdb_gen_df with imdb_gen_df2

# imdb_gen_df.drop(columns=imdb_gen_df2.columns, inplace=True)
# imdb_gen_df = pd.concat([imdb_gen_df, imdb_gen_df2], axis=1)

# New quantile plot


# print(imdb_gen_df.columns)
# a = input("Press Enter to continue...")

quantile_resolution = 1

quantile_df = imdb_gen_df.copy()
quantile_df = quantile_df.sort_values(by=["prompt_score"])
quantile_df = quantile_df.reset_index(drop=True)

# Stuff to plot
gpt2_quantiles = []
rlhf_quantiles = []
rlhf_risk_quantiles = []
sft_quantiles = []

for quantile_point in range(1, 101, quantile_resolution):
    # selecct first quantile_point% of the data
    relevant_df = quantile_df.iloc[: int((quantile_point / 100) * quantile_df.shape[0])]

    gpt2_quantiles.append(relevant_df["R_gpt2"].mean())
    rlhf_quantiles.append(relevant_df["R"].mean())
    rlhf_risk_quantiles.append(relevant_df["R_risk"].mean())
    sft_quantiles.append(relevant_df["R_sft"].mean())

# Plot
fig, axs = plt.subplots(figsize=(10, 5))

axs.plot(range(1, 101, quantile_resolution), gpt2_quantiles, label="GPT2", color="red")
axs.plot(range(1, 101, quantile_resolution), rlhf_quantiles, label="RLHF", color="blue")
axs.plot(range(1, 101, quantile_resolution), rlhf_risk_quantiles, label="RA-RLHF", color="green")
axs.plot(range(1, 101, quantile_resolution), sft_quantiles, label="SFT", color="orange")

axs.set_xlabel("Quantile (%)", fontsize=16)  # Set label size on the axis object
axs.set_ylabel(f"Average Reward", fontsize=16)
axs.tick_params(axis="both", which="major", labelsize=14)
axs.grid(True)  # Add grid on the axis object
axs.legend(loc="upper left")
axs.set_title(f"Average Reward vs Quantile")

plt.tight_layout()
plt.savefig("new_quantile_plot_imdb.png")
plt.savefig("new_quantile_plot_imdb.pdf")
print("Verify plot saved")
print("GPT2: ", gpt2_quantiles[-1])
print("RLHF: ", rlhf_quantiles[-1])
print("RA-RLHF: ", rlhf_risk_quantiles[-1])
print("SFT: ", sft_quantiles[-1])
