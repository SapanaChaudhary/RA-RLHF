import matplotlib.pyplot as plt

# import plotly.express as px
import pandas as pd
import numpy as np
import pdb

# IMDB
imdb_df_test = pd.read_csv("jigsaw_generations_35_32.csv")

# tail_to_save = -2.8
# tail_df = imdb_df_test[imdb_df_test['prompt_score'] <= tail_to_save]
# tail_df = tail_df.drop(columns=['input_ids'])
# print(tail_df['R'].mean())
# print(tail_df['R_risk'].mean())
# tail_df.to_csv("tail/imdb_tail.csv", index=False)
# pdb.set_trace()

# Rename as needed in plots
# imdb_df_test = imdb_df_test.rename(columns={"R": "RLHF", "R_risk": "RLHF risk averse"})
# imdb_df_test = imdb_df_test.rename(columns={"R_bar": "RLHF returns", "R_bar_risk": "RLHF risk averse returns"})


# # CDF plots
# fig = px.ecdf(
#     imdb_df_test,
#     x=["RLHF", "RLHF risk averse"],
#     labels={
#         "value": "Reward",
#         "variable": "Model",
#     },
# )
# fig.update_layout(
#     title="Reward distribution for RLHF and RLHF risk averse models",
#     xaxis_title="Reward",
#     yaxis_title="CDF",
# )
# print("Average reward for RLHF: ", imdb_df_test["RLHF"].mean())
# print("Average reward for RLHF risk averse: ", imdb_df_test["RLHF risk averse"].mean())
# fig.write_image("imdb_R_cdf.png")

# fig = px.ecdf(
#     imdb_df_test,
#     x=["RLHF returns", "RLHF risk averse returns"],
#     labels={
#         "value": "Reward",
#         "variable": "Model",
#     },
# )
# fig.update_layout(
#     title="Returns distribution for RLHF and RLHF risk averse models",
#     xaxis_title="Returns",
#     yaxis_title="CDF",
# )
# print("Average return for RLHF: ", imdb_df_test["RLHF returns"].mean())
# print("Average return for RLHF risk averse: ", imdb_df_test["RLHF risk averse returns"].mean())
# fig.write_image("imdb_R_bar_cdf.png")

# pdb.set_trace()
# Histogram shift plots for imdb


# Function to plot only the top edges of a histogram
def plot_hist_top_edges(data, ax, bins, color, label):
    hist, bin_edges = np.histogram(data, bins=bins)
    for left, right, height in zip(bin_edges[:-1], bin_edges[1:], hist):
        ax.plot([left, right], [height, height], color=color, label=label)
        label = None  # To avoid duplicate labels in the legend


fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(30, 6))
word_map = {"prompt_score": "Prompt Score", "R": "RLHF", "R_risk": "RA-RLHF"}
for i, col in enumerate(["prompt_score", "R", "R_risk"]):
    imdb_df_test[imdb_df_test["toxic"] == 1][col].plot.hist(ax=axs[i], bins=30, alpha=0.5, label="Toxic")
    imdb_df_test[imdb_df_test["toxic"] == 0][col].plot.hist(ax=axs[i], bins=30, alpha=0.5, label="Non Toxic")

    axs[i].set_title(f"{word_map[col]}", fontsize=30)
    axs[i].set_xlabel("Reward", fontsize=30)  # Set label size on the axis object
    axs[i].set_ylabel("")
    axs[i].tick_params(axis="both", which="major", labelsize=28)
    axs[i].grid(True)  # Add grid on the axis object
    axs[i].legend(loc="upper left", fontsize=24)  # Add legend on the axis objec


# Find the maximum y-value among the first three histograms
max_count = max(axs[0].get_ylim()[1], axs[1].get_ylim()[1], axs[2].get_ylim()[1])

# Set the y-axis limits for the first three subplots to be the same
axs[0].set_ylim(0, max_count)
axs[1].set_ylim(0, max_count)
axs[2].set_ylim(0, max_count)
axs[0].set_ylabel(f"Count", fontsize=30)

# tail prompts
tail_filter_level = -6.0
filtered_df = imdb_df_test[imdb_df_test["prompt_score"] <= tail_filter_level]
print(filtered_df.shape[0])
filtered_df[filtered_df["toxic"] == 0]["prompt_score"].plot.hist(ax=axs[3], bins=30, alpha=0.5, label="Non Toxic")
filtered_df[filtered_df["toxic"] == 1]["prompt_score"].plot.hist(ax=axs[3], bins=30, alpha=0.5, label="Toxic")
axs[3].set_title("Tail Prompts", fontsize=30)
axs[3].set_xlabel("Reward", fontsize=30)  # Set label size on the axis object
axs[3].set_ylabel("")
axs[3].tick_params(axis="both", which="major", labelsize=28)
axs[3].grid(True)  # Add grid on the axis object
axs[3].legend(loc="upper left", fontsize=24)  # Add legend on the axis objec
axs[3].axvline(x=tail_filter_level, color="red", linestyle="-", linewidth=4)

# tail performance
filtered_df["R"].plot.hist(ax=axs[4], bins=30, alpha=0.3, label="RLHF", color="red")
filtered_df["R_risk"].plot.hist(ax=axs[4], bins=30, alpha=0.5, label="RA-RLHF", color="olive")
axs[4].set_title("Tail Performance", fontsize=30)
axs[4].set_xlabel("Reward", fontsize=30)  # Set label size on the axis object
axs[4].set_ylabel("")
axs[4].tick_params(axis="both", which="major", labelsize=28)
axs[4].grid(True)  # Add grid on the axis object
axs[4].legend(loc="upper right", fontsize=24)  # Add legend on the axis objec
axs[4].axvline(x=tail_filter_level, color="red", linestyle="-", linewidth=4)

# Find the maximum y-value among the first three histograms
max_count_2 = max(axs[3].get_ylim()[1], axs[4].get_ylim()[1])

# Set the y-axis limits for the first three subplots to be the same
axs[3].set_ylim(0, max_count_2)
axs[4].set_ylim(0, max_count_2)

fig.tight_layout()
plt.savefig("./test_plots_imdb/imdb_distribution_shifts.png")
plt.savefig("./test_plots_imdb/imdb_distribution_shifts.pdf")
plt.clf()

# Quantile plots of returns for imdb

fig, axs = plt.subplots(figsize=(10, 5))

word_map = {"R_bar": "RLHF returns", "R_bar_risk": "RA-RLHF returns"}

for i, col in enumerate(["R_bar", "R_bar_risk"]):
    x = np.linspace(0, 1, len(imdb_df_test[col])) * 100
    y = imdb_df_test[col].sort_values().values
    axs.plot(x, y, label=word_map[col])

axs.set_xlabel("Quantile (%)", fontsize=16)  # Set label size on the axis object
axs.set_ylabel(f"Returns", fontsize=16)
axs.tick_params(axis="both", which="major", labelsize=14)
axs.grid(True)  # Add grid on the axis object
axs.legend(loc="upper left")
axs.set_title(f"Quantile plot of returns")


fig.tight_layout()
plt.savefig("./test_plots_imdb/imdb_R_bar_quantiles.png")
plt.savefig("./test_plots_imdb/imdb_R_bar_quantiles.pdf")
plt.clf()


# Quantile plots of rewards for imdb

fig, axs = plt.subplots(figsize=(10, 5))

word_map = {"R": "RLHF", "R_risk": "RA-RLHF"}

for i, col in enumerate(["R", "R_risk"]):
    x = np.linspace(0, 1, len(imdb_df_test[col])) * 100
    y = imdb_df_test[col].sort_values().values
    axs.plot(x, y, label=word_map[col])

axs.set_xlabel("Quantile (%)", fontsize=16)  # Set label size on the axis object
axs.set_ylabel(f"Rewards", fontsize=16)
axs.tick_params(axis="both", which="major", labelsize=14)
axs.grid(True)  # Add grid on the axis object
axs.legend(loc="upper left")
axs.set_title(f"Quantile plot of rewards")


fig.tight_layout()
plt.savefig("./test_plots_imdb/imdb_R_quantiles.png")
plt.savefig("./test_plots_imdb/imdb_R_quantiles.pdf")
plt.clf()

# # Box plots of returns for imdb

# fig, axs = plt.subplots(figsize=(10, 5))


# # # Jigsaw
# jigsaw_df_test = pd.read_csv("jigsaw_generations.csv")

# # Rename as needed in plots
# # jigsaw_df_test = jigsaw_df_test.rename(columns={"R": "RLHF", "R_risk": "RLHF risk averse"})
# # jigsaw_df_test = jigsaw_df_test.rename(columns={"R_bar": "RLHF returns", "R_bar_risk": "RLHF risk averse returns"})

# # # CDF plots
# # fig = px.ecdf(
# #     jigsaw_df_test,
# #     x=["RLHF", "RLHF risk averse"],
# #     labels={
# #         "value": "Reward",
# #         "variable": "Model",
# #     },
# # )
# # fig.update_layout(
# #     title="Reward distribution for RLHF and RLHF risk averse models",
# #     xaxis_title="Reward",
# #     yaxis_title="CDF",
# # )
# # print("Average reward for RLHF: ", jigsaw_df_test["RLHF"].mean())
# # print("Average reward for RLHF risk averse: ", jigsaw_df_test["RLHF risk averse"].mean())
# # fig.write_image("jigsaw_R_cdf.png")

# # fig = px.ecdf(
# #     jigsaw_df_test,
# #     x=["RLHF returns", "RLHF risk averse returns"],
# #     labels={
# #         "value": "Reward",
# #         "variable": "Model",
# #     },
# # )
# # fig.update_layout(
# #     title="Returns distribution for RLHF and RLHF risk averse models",
# #     xaxis_title="Returns",
# #     yaxis_title="CDF",
# # )
# # print("Average return for RLHF: ", jigsaw_df_test["RLHF returns"].mean())
# # print("Average return for RLHF risk averse: ", jigsaw_df_test["RLHF risk averse returns"].mean())
# # fig.write_image("jigsaw_R_bar_cdf.png")

# # Plot histogram shifts for jigsaw
# fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
# word_map = {"prompt_score": "Prompt", "R": "RLHF", "R_risk": "RA-RLHF"}
# for i, col in enumerate(["prompt_score", "R", "R_risk"]):
#     jigsaw_df_test[jigsaw_df_test["toxic"] == 0][col].plot.hist(ax=axs[i], bins=30, alpha=0.5, label="Non Toxic")
#     jigsaw_df_test[jigsaw_df_test["toxic"] == 1][col].plot.hist(ax=axs[i], bins=30, alpha=0.5, label="Toxic")

#     axs[i].set_xlabel("Reward", fontsize=16)  # Set label size on the axis object
#     axs[i].set_ylabel(f"Count", fontsize=16)
#     axs[i].tick_params(axis="both", which="major", labelsize=14)
#     axs[i].grid(True)  # Add grid on the axis object
#     axs[i].legend(loc="upper left")  # Add legend on the axis objec
#     axs[i].set_title(f"{word_map[col]}")


# fig.tight_layout()
# plt.savefig("./test_plots_jigsaw/jigsaw_distribution_shifts.png")
# plt.savefig("./test_plots_jigsaw/jigsaw_distribution_shifts.pdf")
# plt.clf()

# # Quantile plots of returns for jigsaw

# fig, axs = plt.subplots(figsize=(10, 5))
# word_map = {"R_bar": "RLHF returns", "R_bar_risk": "RA-RLHF returns"}

# for i, col in enumerate(["R_bar", "R_bar_risk"]):
#     x = np.linspace(0, 1, len(jigsaw_df_test[col])) * 100
#     y = jigsaw_df_test[col].sort_values().values
#     axs.plot(x, y, label=word_map[col])

# axs.set_xlabel("Quantile (%)", fontsize=16)  # Set label size on the axis object
# axs.set_ylabel(f"Returns", fontsize=16)
# axs.tick_params(axis="both", which="major", labelsize=14)
# axs.grid(True)  # Add grid on the axis object
# axs.legend(loc="upper left")
# axs.set_title(f"Quantile plot of returns")


# fig.tight_layout()
# plt.savefig("./test_plots_jigsaw/jigsaw_R_bar_quantiles.png")
# plt.savefig("./test_plots_jigsaw/jigsaw_R_bar_quantiles.pdf")
# plt.clf()

# # Quantile plots of rewards for jigsaw

# fig, axs = plt.subplots(figsize=(10, 5))
# word_map = {"R": "RLHF", "R_risk": "RA-RLHF"}
# for i, col in enumerate(["R", "R_risk"]):
#     x = np.linspace(0, 1, len(jigsaw_df_test[col])) * 100
#     y = jigsaw_df_test[col].sort_values().values
#     axs.plot(x, y, label=word_map[col])

# axs.set_xlabel("Quantile (%)", fontsize=16)  # Set label size on the axis object
# axs.set_ylabel(f"Rewards", fontsize=16)
# axs.tick_params(axis="both", which="major", labelsize=14)
# axs.grid(True)  # Add grid on the axis object
# axs.legend(loc="upper left")
# axs.set_title(f"Quantile plot of rewards")


# fig.tight_layout()
# plt.savefig("./test_plots_jigsaw/jigsaw_R_quantiles.png")
# plt.savefig("./test_plots_jigsaw/jigsaw_R_quantiles.pdf")
# plt.clf()
