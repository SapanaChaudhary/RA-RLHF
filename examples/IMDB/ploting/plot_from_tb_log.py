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
    scalar_tags = ea.Tags()['scalars']

    # Dictionary to hold the data
    scalar_data = {}

    # Extract data for each scalar tag
    for tag in scalar_tags:
        events = ea.Scalars(tag)
        times = [e.wall_time for e in events]
        steps = [e.step for e in events]
        values = [e.value for e in events]
        scalar_data[tag] = {'times': times, 'steps': steps, 'values': values}

    return scalar_data

# Path to your TensorBoard log file or directory
log_file_path = '/mnt/shared-scratch/Anon1/auth1/rlhf/trl/logs/dummy/2024-01-18_01-23-52/trl'  # Replace with the actual path to your log file

# Extract data
data = extract_scalar_data_from_event_file(log_file_path)

# Assuming 'data' is a list of dictionaries loaded from multiple log files
# and each dictionary in 'data' has the same structure as what we've discussed previously.

# Step 1: Group data by tag
tag_grouped_data = {}
for single_run_data in data:
    for tag, values in single_run_data.items():
        if tag not in tag_grouped_data:
            tag_grouped_data[tag] = []
        tag_grouped_data[tag].append(values)

# Step 2: Calculate mean and standard deviation for each tag
tag_stats = {}
for tag, values_list in tag_grouped_data.items():
    # Assuming that all runs have the same number of steps and are aligned
    steps = values_list[0]['steps']
    all_values = np.array([values['values'] for values in values_list])
    mean_values = np.mean(all_values, axis=0)
    std_values = np.std(all_values, axis=0)
    tag_stats[tag] = {'steps': steps, 'mean': mean_values, 'std': std_values}

# Step 3: Plot mean and standard deviation for each tag
for tag, stats in tag_stats.items():
    plt.figure(figsize=(10, 5))
    steps = stats['steps']
    mean_values = stats['mean']
    std_values = stats['std']
    plt.plot(steps, mean_values, label=f'{tag} mean')
    plt.fill_between(steps, mean_values - std_values, mean_values + std_values, alpha=0.2, label=f'{tag} std dev')
    plt.xlabel('Steps')
    plt.ylabel(tag)
    plt.title(f'Mean and Standard Deviation for {tag}')
    plt.legend()
    plt.show()
