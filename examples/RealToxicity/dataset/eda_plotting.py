from datasets import load_dataset, load_from_disk
from matplotlib import pyplot as plt
import pandas as pd

real_toxicity = load_from_disk(
    "/mnt/research/Anon2/Students/auth2/datasets/RealToxicity/real_toxicity_test.hf"
)


def plot_histogram(dataset, feature_name):
    fig, ax = plt.subplots()
    dataset_label_0 = dataset.filter(lambda x: x["Label"] == 0)
    dataset_label_1 = dataset.filter(lambda x: x["Label"] == 1)
    dataset_label_0 = pd.Series(dataset_label_0[feature_name])
    dataset_label_1 = pd.Series(dataset_label_1[feature_name])
    # print(dataset_label_0)
    dataset_label_0.plot.hist(ax=ax, bins=40, alpha=0.5, label="Non Toxic")
    dataset_label_1.plot.hist(ax=ax, bins=40, alpha=0.5, label="Toxic")
    ax.set_title(f"Histogram of {feature_name} - RealToxicity Test")
    ax.legend()
    plt.savefig(f"{feature_name}_histogram_RealToxicitytest.png")
    plt.clf()


# Perspective
plot_histogram(real_toxicity, "Perspective")

# Unitary
plot_histogram(real_toxicity, "Unitary")

# Facebook
plot_histogram(real_toxicity, "Facebook")
