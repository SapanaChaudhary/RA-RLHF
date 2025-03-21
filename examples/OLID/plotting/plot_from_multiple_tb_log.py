import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

import pdb


# Function to extract scalar data from a TensorBoard log file
def extract_scalar_data_from_event_file(event_file_path):
    # Initialize an accumulator
    ea = event_accumulator.EventAccumulator(event_file_path)
    ea.Reload()  # Loads the log data from file

    # Get all scalar tags
    scalar_tags = ea.Tags()["scalars"]

    # Dictionary to hold the data
    scalar_data = {}

    # Extract data for each scalar tag
    for tag in scalar_tags:
        events = ea.Scalars(tag)
        times = [e.wall_time for e in events]
        steps = [e.step for e in events]
        values = [e.value for e in events]
        scalar_data[tag] = {"times": times, "steps": steps, "values": values}

    return scalar_data


# Function for smoothing the curve (simple moving average for illustration)
def smooth_curve(points, factor=0.97):
    smoothed_points = np.zeros_like(points)
    for i in range(len(points)):
        if i > 0:
            smoothed_points[i] = (smoothed_points[i - 1] * factor) + (points[i] * (1 - factor))
        else:
            smoothed_points[i] = points[i]
    return smoothed_points


# Path to your TensorBoard log file or directory
# Assuming `all_data` is a structured dictionary containing all runs, organized by algorithm and then by seed
# e.g., all_data = {'algorithm1': {'seed1': data1, 'seed2': data2, ...},
#                   'algorithm2': {'seed1': data1, 'seed2': data2, ...}, ...}

log_file_paths = {
    "RA-RLHF": [
        "/mnt/research/Anon2/Students/auth2/rlhf/trl/logs/dummy/OLID/2024-02-19_01-09-26/trl",
        "/mnt/research/Anon2/Students/auth2/rlhf/trl/logs/dummy/OLID/2024-02-19_01-24-53/trl",
        "/mnt/research/Anon2/Students/auth2/rlhf/trl/logs/dummy/OLID/2024-02-19_01-40-24/trl",
    ],
    "RLHF": [
        "/mnt/research/Anon2/Students/auth2/rlhf/trl/logs/dummy/OLID/2024-02-19_01-08-02/trl",
        "/mnt/research/Anon2/Students/auth2/rlhf/trl/logs/dummy/OLID/2024-02-19_01-27-30/trl",
        "/mnt/research/Anon2/Students/auth2/rlhf/trl/logs/dummy/OLID/2024-02-19_01-46-52/trl",
    ],
}

# IMDB Sanity Check
# log_file_paths = {
#     "RA-RLHF": [
#         "/mnt/research/Anon2/Students/auth2/rlhf/trl/logs/dummy/IMDB/2024-02-18_23-09-57/trl",
#         "/mnt/research/Anon2/Students/auth2/rlhf/trl/logs/dummy/IMDB/2024-02-19_00-16-45/trl",
#         "/mnt/research/Anon2/Students/auth2/rlhf/trl/logs/dummy/IMDB/2024-02-19_01-24-12/trl",
#     ],
#     "RLHF": [
#         "/mnt/research/Anon2/Students/auth2/rlhf/trl/logs/dummy/IMDB/2024-02-19_01-15-25/trl",
#         "/mnt/research/Anon2/Students/auth2/rlhf/trl/logs/dummy/IMDB/2024-02-19_02-24-58/trl",
#         "/mnt/research/Anon2/Students/auth2/rlhf/trl/logs/dummy/IMDB/2024-02-19_03-36-47/trl",
#     ],
# }

# Extract data
all_data = {}
for algorithm, paths in log_file_paths.items():
    all_data[algorithm] = {}
    for i, path in enumerate(paths):
        seed_key = f"seed{i+1}"
        # load data
        all_data[algorithm][seed_key] = extract_scalar_data_from_event_file(path)

# all_data['RLHF']['seed1']['objective/kl'].keys()
# dict_keys(['times', 'steps', 'values'])

desired_tags = {
    "env/reward_mean": "Environment Reward",
    "env/reward_std": "Reward std",
    "objective/entropy": "Policy Entropy",
    "objective/kl": "KL Divergence",
    "objective/kl_coef": "Beta",
    "ppo/returns/mean": "Return over Batch",
    "tokens/responses_len_mean": "Response Length",
}

# Step 1: Calculate mean and standard deviation for each tag for each algorithm
tag_stats_by_algorithm = {}
for algorithm, seeds_data in all_data.items():
    tag_grouped_data = {}
    for seed, data in seeds_data.items():
        print(seed)
        for tag, values in data.items():
            # pdb.set_trace()
            if tag in desired_tags.keys():
                print(tag)
                if tag not in tag_grouped_data:
                    # pdb.set_trace()
                    print("here")
                    tag_grouped_data[tag] = []
                tag_grouped_data[tag].append(values)

    # pdb.set_trace()
    tag_stats = {}
    for tag, values_list in tag_grouped_data.items():
        # Assuming all seed runs have the same number of steps and are aligned
        steps = values_list[0]["steps"]
        all_values = np.array([values["values"] for values in values_list])
        mean_values = np.mean(all_values, axis=0)
        std_values = np.std(all_values, axis=0)
        tag_stats[tag] = {"steps": steps, "mean": mean_values, "std": std_values}

    tag_stats_by_algorithm[algorithm] = tag_stats


# Step 2: Plot mean and standard deviation for each tag for each algorithm
for tag in tag_grouped_data.keys():
    fig, ax = plt.subplots(figsize=(5, 5))  # Create a figure and an axis object

    for algorithm, stats in tag_stats_by_algorithm.items():
        if tag in stats:
            print(tag)
            steps = stats[tag]["steps"]
            mean_values = smooth_curve(stats[tag]["mean"])
            std_values = smooth_curve(stats[tag]["std"])

            if algorithm == "RLHF":
                ax.plot(steps, mean_values, label=f"{algorithm}", color="red")
                ax.fill_between(
                    steps, mean_values - std_values, mean_values + std_values, color="red", alpha=0.2
                )  # , label=f'{algorithm} std dev')
            else:
                ax.plot(steps, mean_values, label=f"{algorithm}", color="green")
                ax.fill_between(
                    steps, mean_values - std_values, mean_values + std_values, color="green", alpha=0.2
                )  # , label=f'{algorithm} std dev')
            ax.set_xlabel("Training iteration", fontsize=24)  # Set label size on the axis object
            ax.set_ylabel(f"{desired_tags[tag]}", fontsize=24)  # Set label size on the axis object
            ax.tick_params(axis="both", which="major", labelsize=20)  # Adjust tick size on the axis object
            # ax.set_box_aspect(0.7)  # Commented out because this might not be available depending on your matplotlib version
            ax.grid(True)  # Add grid on the axis object
            ax.legend(loc="upper left", fontsize=16)  # Add legend on the axis object
            # plt.title(f'Mean and Standard Deviation for {tag} across Algorithms')
            plt.savefig(
                f"train_plots_olid/{desired_tags[tag].replace(' ','_')}.pdf", bbox_inches="tight", pad_inches=0
            )  # as png
