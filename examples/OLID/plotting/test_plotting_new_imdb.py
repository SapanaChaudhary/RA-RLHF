import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np

# # IMDB
# jigsaw_df_test = pd.read_csv("imdb_generations.csv")

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
# # fig.write_image("imdb_R_cdf.png")

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
# # fig.write_image("imdb_R_bar_cdf.png")


# # Histogram shift plots for imdb
# fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
# word_map = {"prompt_score": "Prompt", "R": "RLHF", "R_risk": "RA-RLHF"}
# for i, col in enumerate(["prompt_score", "R", "R_risk"]):
#     jigsaw_df_test[jigsaw_df_test["label"] == 0][col].plot.hist(ax=axs[i], bins=30, alpha=0.5, label="Negative")
#     jigsaw_df_test[jigsaw_df_test["label"] == 1][col].plot.hist(ax=axs[i], bins=30, alpha=0.5, label="Positive")

#     axs[i].set_title(f"{word_map[col]}")
#     axs[i].set_xlabel("Reward", fontsize=16)  # Set label size on the axis object
#     axs[i].set_ylabel(f"Count", fontsize=16)
#     axs[i].tick_params(axis="both", which="major", labelsize=14)
#     axs[i].grid(True)  # Add grid on the axis object
#     axs[i].legend(loc="upper left")  # Add legend on the axis objec

# fig.tight_layout()
# plt.savefig("./test_plots_imdb/imdb_distribution_shifts.png")
# plt.savefig("./test_plots_imdb/imdb_distribution_shifts.pdf")
# plt.clf()

# # Quantile plots of returns for imdb

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
# plt.savefig("./test_plots_imdb/imdb_R_bar_quantiles.png")
# plt.savefig("./test_plots_imdb/imdb_R_bar_quantiles.pdf")
# plt.clf()


# # Quantile plots of rewards for imdb

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
# plt.savefig("./test_plots_imdb/imdb_R_quantiles.png")
# plt.savefig("./test_plots_imdb/imdb_R_quantiles.pdf")
# plt.clf()

# # Box plots of returns for imdb

# fig, axs = plt.subplots(figsize=(10, 5))


# # Jigsaw
jigsaw_df_test = pd.read_csv(
    "/mnt/research/Anon2/Students/auth2/repos/trl/examples/OLID/dataset/imdb_generations_64_48_alpha_40_seed_4_gptj.csv"
)
# jigsaw_df_test_2 = pd.read_csv("../dataset/jigsaw_generations_8_32_alpha_40_seed_2.csv")

# jigsaw_df_test_2.drop(columns=jigsaw_df_test.columns, inplace=True)
# jigsaw_df_test = pd.concat([jigsaw_df_test, jigsaw_df_test_2], axis=1)


# Rename as needed in plots
# jigsaw_df_test = jigsaw_df_test.rename(columns={"R": "RLHF", "R_risk": "RLHF risk averse"})
# jigsaw_df_test = jigsaw_df_test.rename(columns={"R_bar": "RLHF returns", "R_bar_risk": "RLHF risk averse returns"})

# # CDF plots
# fig = px.ecdf(
#     jigsaw_df_test,
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
# print("Average reward for RLHF: ", jigsaw_df_test["RLHF"].mean())
# print("Average reward for RLHF risk averse: ", jigsaw_df_test["RLHF risk averse"].mean())
# fig.write_image("jigsaw_R_cdf.png")

# fig = px.ecdf(
#     jigsaw_df_test,
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
# print("Average return for RLHF: ", jigsaw_df_test["RLHF returns"].mean())
# print("Average return for RLHF risk averse: ", jigsaw_df_test["RLHF risk averse returns"].mean())
# fig.write_image("jigsaw_R_bar_cdf.png")

# tail_to_save = -3

# tail_df = jigsaw_df_test[jigsaw_df_test["prompt_score"] <= tail_to_save]

# tail_df = tail_df.drop(columns=["input_ids"])

# print(tail_df["R"].mean())

# print(tail_df["R_risk"].mean())

zoom_border_color = "limegreen"


# Plot histogram shifts for jigsaw
def plot_hist_top_edges(data, ax, bins, color, label):
    hist, bin_edges = np.histogram(data, bins=bins)

    for left, right, height in zip(bin_edges[:-1], bin_edges[1:], hist):
        ax.plot([left, right], [height, height], color=color, label=label)

        label = None  # To avoid duplicate labels in the legend


fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(30, 6))

word_map = {
    "prompt_score": "Prompt Score",
    "R": "RLHF",
    "R_risk": "RA-RLHF",
    "R_sft": "SFT",
}

for i, col in enumerate(["prompt_score", "R_sft", "R", "R_risk"]):
    jigsaw_df_test[jigsaw_df_test["label"] == 0][col].plot.hist(ax=axs[i], bins=30, alpha=0.5, label="Not Offensive")
    jigsaw_df_test[jigsaw_df_test["label"] == 1][col].plot.hist(ax=axs[i], bins=30, alpha=0.5, label="Offensive")

    axs[i].set_title(f"{word_map[col]}", fontsize=30)

    axs[i].set_xlabel("Reward", fontsize=30)  # Set label size on the axis object

    axs[i].set_ylabel("")

    axs[i].tick_params(axis="both", which="major", labelsize=28)

    axs[i].grid(True)  # Add grid on the axis object

    axs[i].legend(loc="upper left", fontsize=24)  # Add legend on the axis objec

    # # zoomed in bit
    # left_lim = -1.3
    # right_lim = 0
    # axs_zoom = axs[i].inset_axes([0.10, 0.20, 0.65, 0.40])  # left, bottom, width, height

    # axs_zoom.hist(
    #     jigsaw_df_test[jigsaw_df_test["label"] == 0][col],
    #     bins=30,
    #     range=[left_lim, right_lim],
    #     alpha=0.5,
    #     label="Not Offensive",
    # )
    # axs_zoom.hist(
    #     jigsaw_df_test[jigsaw_df_test["label"] == 1][col],
    #     bins=30,
    #     range=[left_lim, right_lim],
    #     alpha=0.5,
    #     label="Offensive",
    # )
    # axs_zoom.set_ylim(0, 10)
    # axs_zoom.set_xlim(left_lim, right_lim)
    # axs_zoom.xaxis.set_major_locator(plt.MaxNLocator(3))
    # axs_zoom.yaxis.set_major_locator(plt.MaxNLocator(3))
    # axs_zoom.tick_params(axis="both", which="major", labelsize=20)
    # # set border color to orange
    # axs_zoom.spines["bottom"].set_color(zoom_border_color)
    # axs_zoom.spines["top"].set_color(zoom_border_color)
    # axs_zoom.spines["left"].set_color(zoom_border_color)
    # axs_zoom.spines["right"].set_color(zoom_border_color)
    # # set border width to zoom_thickness
    # zoom_thickness = 3.0
    # axs_zoom.spines["bottom"].set_linewidth(zoom_thickness)
    # axs_zoom.spines["top"].set_linewidth(zoom_thickness)
    # axs_zoom.spines["left"].set_linewidth(zoom_thickness)
    # axs_zoom.spines["right"].set_linewidth(zoom_thickness)
    # # rectangle coords in zoomed in plot
    # rect_height = 15
    # rectangle = plt.Rectangle(
    #     (left_lim, 0),
    #     right_lim - left_lim,
    #     rect_height,
    #     facecolor="none",
    #     edgecolor=zoom_border_color,
    #     lw=zoom_thickness,
    # )
    # axs[i].add_patch(rectangle)


# Find the maximum y-value among the first three histograms

max_count = max(axs[0].get_ylim()[1], axs[1].get_ylim()[1], axs[2].get_ylim()[1])


# Set the y-axis limits for the first three subplots to be the same
axs[0].set_ylim(0, max_count)
axs[1].set_ylim(0, max_count)
axs[2].set_ylim(0, max_count)
axs[3].set_ylim(0, max_count)
axs[0].set_ylabel(f"Count", fontsize=30)

# make quantile plot
# imdb_gen_df = imdb_df_test2
# imdb_gen_df2 = imdb_df_test
# imdb_gen_df.drop(columns=imdb_gen_df2.columns, inplace=True)
# imdb_gen_df = pd.concat([imdb_gen_df, imdb_gen_df2], axis=1)

quantile_resolution = 1

quantile_df = jigsaw_df_test.copy()
quantile_df = quantile_df.sort_values(by=["prompt_score"])
quantile_df = quantile_df.reset_index(drop=True)

# Stuff to plot
gpt2_quantiles = []
gptj_quantiles = []
rlhf_quantiles = []
rlhf_risk_quantiles = []
sft_quantiles = []

for quantile_point in range(1, 101, quantile_resolution):
    # select first quantile_point% of the data
    relevant_df = quantile_df.iloc[: int((quantile_point / 100) * quantile_df.shape[0])]

    gpt2_quantiles.append(relevant_df["R_gpt2"].mean())
    gptj_quantiles.append(relevant_df["R_gptj"].mean())
    rlhf_quantiles.append(relevant_df["R"].mean())
    rlhf_risk_quantiles.append(relevant_df["R_risk"].mean())
    sft_quantiles.append(relevant_df["R_sft"].mean())

axs[4].plot(range(1, 101, quantile_resolution), gpt2_quantiles, label="GPT2", color="black")
axs[4].plot(range(1, 101, quantile_resolution), gptj_quantiles, label="GPTJ", color="pink")
axs[4].plot(range(1, 101, quantile_resolution), sft_quantiles, label="SFT", color="blue")
axs[4].plot(range(1, 101, quantile_resolution), rlhf_quantiles, label="RLHF", color="red")
axs[4].plot(
    range(1, 101, quantile_resolution),
    rlhf_risk_quantiles,
    label="RA-RLHF",
    color="green",
)


axs[4].set_xlabel("Quantile (%)", fontsize=30)  # Set label size on the axis object
axs[4].set_ylabel(f"Average Reward", fontsize=30)
# axs[4].set_yscale("log")
axs[4].tick_params(axis="both", which="major", labelsize=28)
axs[4].grid(True)  # Add grid on the axis object
axs[4].legend(loc="lower right", fontsize=16)
axs[4].set_title(f"Reward vs Quantile", fontsize=30)


# # Zoomed in bit for the quantile plot
# axs_zoom = axs[4].inset_axes([0.65, 0.42, 0.30, 0.30])
# axs_zoom.plot(range(1, 101, quantile_resolution), gpt2_quantiles, label="GPT2", color="black")
# axs_zoom.plot(range(1, 101, quantile_resolution), sft_quantiles, label="SFT", color="blue")
# axs_zoom.plot(range(1, 101, quantile_resolution), rlhf_quantiles, label="RLHF", color="red")
# axs_zoom.plot(range(1, 101, quantile_resolution), rlhf_risk_quantiles, label="RA-RLHF", color="green")
# # increase tick font size
# axs_zoom.tick_params(axis="both", which="major", labelsize=18)
# # set border color to orange
# axs_zoom.spines["bottom"].set_color(zoom_border_color)
# axs_zoom.spines["top"].set_color(zoom_border_color)
# axs_zoom.spines["left"].set_color(zoom_border_color)
# axs_zoom.spines["right"].set_color(zoom_border_color)
# # set border width to zoom_thickness
# zoom_thickness = 3.0
# axs_zoom.spines["bottom"].set_linewidth(zoom_thickness)
# axs_zoom.spines["top"].set_linewidth(zoom_thickness)
# axs_zoom.spines["left"].set_linewidth(zoom_thickness)
# axs_zoom.spines["right"].set_linewidth(zoom_thickness)


# # rectangle coords in zoomed in plot
# x1, x2, y1, y2 = 0.3, 10.0, -1.2, -0.8
# axs_zoom.set_xlim(x1, x2)
# axs_zoom.set_ylim(y1, y2)
# rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor="none", edgecolor=zoom_border_color, lw=zoom_thickness)
# axs[4].add_patch(rect)

fig.tight_layout()
plt.savefig("./test_plots_imdb/imdb_distribution_shifts.png")
plt.savefig("./test_plots_imdb/imdb_distribution_shifts.pdf")
plt.clf()


print("\n\n")
print("Average reward for RLHF: ", jigsaw_df_test["R"].mean())
print("Average reward for RA-RLHF: ", jigsaw_df_test["R_risk"].mean())
print("Average reward for SFT: ", jigsaw_df_test["R_sft"].mean())
print("Average reward for GPTJ: ", jigsaw_df_test["R_gptj"].mean())
print("Average reward for GPT2:", jigsaw_df_test["R_gpt2"].mean())
# print("Average Perplexity for RLHF: ", jigsaw_df_test["perplexity"].mean())
# print("Average Perplexity for RA-RLHF: ", jigsaw_df_test["perplexity_risk"].mean())
# print("Average Perplexity for SFT: ", jigsaw_df_test["perplexity_sft"].mean())
# print("Average Perplexity for GPT2: ", jigsaw_df_test["perplexity_gpt2"].mean())
print("\n\n")

filtered_df = jigsaw_df_test[jigsaw_df_test["prompt_score"] < -2.5]
print("Tail Average reward for RLHF: ", filtered_df["R"].mean())
print("Tail Average reward for RA-RLHF: ", filtered_df["R_risk"].mean())
print("Tail Average reward for SFT: ", filtered_df["R_sft"].mean())
print("Tail Average reward for GPT2: ", filtered_df["R_gpt2"].mean())
print("Tail Average reward for GPTJ: ", filtered_df["R_gptj"].mean())
# print("Tail Average Perplexity for RLHF: ", filtered_df["perplexity"].mean())
# print("Tail Average Perplexity for RA-RLHF: ", filtered_df["perplexity_risk"].mean())
# print("Tail Average Perplexity for SFT: ", filtered_df["perplexity_sft"].mean())
# print("Tail Average Perplexity for GPT2: ", filtered_df["perplexity_gpt2"].mean())


# Quantile plots of returns for jigsaw

# rlhf_returns_quantiles = []

# rlhf_risk_returns_quantiles = []

# for quantile_point in range(1, 101, quantile_resolution):
#     # select first quantile_point% of the data
#     relevant_df = quantile_df.iloc[: int((quantile_point / 100) * quantile_df.shape[0])]

#     rlhf_returns_quantiles.append(relevant_df["R_bar"].mean())
#     rlhf_risk_returns_quantiles.append(relevant_df["R_bar_risk"].mean())


# fig, axs = plt.subplots(figsize=(10, 5))
# word_map = {"R_bar": "RLHF returns", "R_bar_risk": "RA-RLHF returns"}

# axs.plot(range(1, 101, quantile_resolution), rlhf_returns_quantiles, label="RLHF", color="red")
# axs.plot(range(1, 101, quantile_resolution), rlhf_risk_returns_quantiles, label="RA-RLHF", color="green")

# axs.set_xlabel("Quantile (%)", fontsize=30)  # Set label size on the axis object
# axs.set_ylabel(f"Returns", fontsize=30)
# axs.tick_params(axis="both", which="major", labelsize=14)
# axs.grid(True)  # Add grid on the axis object
# axs.legend(loc="upper left", fontsize=16)
# axs.set_title(f"Quantile plot of returns", fontsize=30)


# fig.tight_layout()
# plt.savefig("./test_plots_imdb/imdb_R_bar_quantiles.png")
# plt.savefig("./test_plots_imdb/imdb_R_bar_quantiles.pdf")
# plt.clf()
