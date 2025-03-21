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

# Function for smoothing the curve (simple moving average for illustration)
def smooth_curve(points, factor=0.9):
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
#ra-rlhf seed 73: 2024-01-19_21-28-11/trl 
#ra-rlhf seed 42: 2024-01-18_09-45-09/trl
#rlhf seed 42: 2024-01-18_01-23-52/trl
#rlhf seed 73: 2024-01-19_23-48-35/trl

log_file_paths = {'RA-RLHF': ['/mnt/shared-scratch/Anon1/auth1/rlhf/trl/logs/dummy/2024-01-19_21-28-11/trl', \
                                '/mnt/shared-scratch/Anon1/auth1/rlhf/trl/logs/dummy/2024-01-18_09-45-09/trl'], 
                'RLHF': ['/mnt/shared-scratch/Anon1/auth1/rlhf/trl/logs/dummy/2024-01-18_01-23-52/trl', \
                                '/mnt/shared-scratch/Anon1/auth1/rlhf/trl/logs/dummy/2024-01-19_23-48-35/trl']
                } 

# Extract data
# Extract data
all_data = {}
for algorithm, paths in log_file_paths.items():
    all_data[algorithm] = {}
    for i, path in enumerate(paths):
        seed_key = f'seed{i+1}'
        # load data
        all_data[algorithm][seed_key] = extract_scalar_data_from_event_file(path)  

#all_data['RLHF']['seed1']['objective/kl'].keys()
#dict_keys(['times', 'steps', 'values'])

# Define a list of tags you want to plot
# desired_tags = ['env/reward_mean', 'env/reward_std', 'objective/entropy', 'objective/kl', 'objective/kl_coef']  # Replace with your actual tags of interest
# desired_tag_names = ['Return', 'reward_std', 'Policy Entropy', 'KL Divergence', 'kl_coeff']

desired_tags = {'env/reward_mean': 'Return', \
                'env/reward_std': 'reward_std', \
                'objective/entropy': 'Policy Entropy', \
                'objective/kl': 'KL Divergence', \
                'objective/kl_coef': 'kl_coeff'}

# Step 1: Calculate mean and standard deviation for each tag for each algorithm
tag_stats_by_algorithm = {}
for algorithm, seeds_data in all_data.items():
    tag_grouped_data = {}
    for seed, data in seeds_data.items():
        print(seed)
        for tag, values in data.items():
            #pdb.set_trace()
            if tag in desired_tags.keys():
                print(tag)
                if tag not in tag_grouped_data:
                    #pdb.set_trace()
                    print('here')
                    tag_grouped_data[tag] = []
                tag_grouped_data[tag].append(values)
    
    #pdb.set_trace()
    tag_stats = {}
    for tag, values_list in tag_grouped_data.items():
        # Assuming all seed runs have the same number of steps and are aligned
        steps = values_list[0]['steps']
        all_values = np.array([values['values'] for values in values_list])
        mean_values = np.mean(all_values, axis=0)
        std_values = np.std(all_values, axis=0)
        tag_stats[tag] = {'steps': steps, 'mean': mean_values, 'std': std_values}
    
    tag_stats_by_algorithm[algorithm] = tag_stats


# # Step 1: Calculate mean and standard deviation for each tag for each algorithm
# tag_stats_by_algorithm = {}
# for algorithm, seeds_data in all_data.items():
#     tag_grouped_data = {}
#     for seed, data in seeds_data.items():
#         for tag, values in data.items():
#             if tag in desired_tags.keys():
#                 if tag not in tag_grouped_data:
#                     tag_grouped_data[desired_tags[tag]] = []
#                 tag_grouped_data[desired_tags[tag]].append(values['values'])  # Store only values for simplicity

#     tag_stats = {}
#     for tag, values_list in tag_grouped_data.items():
#         # Debugging print
#         print(f"{algorithm} - {tag}: Sample values for each seed - {values_list}")

#         # Assuming all seed runs have the same number of steps and are aligned
#         steps = seeds_data[next(iter(seeds_data))][tag]['steps']
#         all_values = np.array(values_list)
#         mean_values = np.mean(all_values, axis=0)
#         std_values = np.std(all_values, axis=0)

#         # Debugging print
#         print(f"{algorithm} - {tag}: Mean values - {mean_values[:5]}")
#         print(f"{algorithm} - {tag}: Std values - {std_values[:5]}")

#         tag_stats[tag] = {'steps': steps, 'mean': mean_values, 'std': std_values}
    
#     tag_stats_by_algorithm[algorithm] = tag_stats


# Step 2: Plot mean and standard deviation for each tag for each algorithm
for tag in tag_grouped_data.keys():
    fig, ax = plt.subplots(figsize=(10, 5))  # Create a figure and an axis object
    
    for algorithm, stats in tag_stats_by_algorithm.items():
        if tag in stats:
            print(tag)
            steps = stats[tag]['steps']
            mean_values = smooth_curve(stats[tag]['mean'])
            std_values = smooth_curve(stats[tag]['std'])
            ax.plot(steps, mean_values, label=f'{algorithm} mean')
            ax.fill_between(steps, mean_values - std_values, mean_values + std_values, alpha=0.3, label=f'{algorithm} std dev')

            ax.set_xlabel('Episode time', fontsize=16)  # Set label size on the axis object
            ax.set_ylabel(f'{tag}', fontsize=16)  # Set label size on the axis object
            ax.tick_params(axis='both', which='major', labelsize=14)  # Adjust tick size on the axis object
            # ax.set_box_aspect(0.7)  # Commented out because this might not be available depending on your matplotlib version
            ax.grid(True)  # Add grid on the axis object
            ax.legend(loc='upper left')  # Add legend on the axis object
            plt.title(f'Mean and Standard Deviation for {tag} across Algorithms')
            plt.show()
            plt.savefig(f'/mnt/shared-scratch/Anon1/auth1/rlhf/trl/examples/IMDB/ploting/final_results/trial/{desired_tags[tag]}', bbox_inches='tight', pad_inches=0) #as png
