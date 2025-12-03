# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the datasets (With correct column names)
# fairness_metrics = pd.read_csv("predictions_output/fairness_metrics_results.csv")
# fairness_metrics_counter = pd.read_csv("predictions_output/fairness_metrics_results_counterfactual.csv")
# fairness_metrics_cot = pd.read_csv("predictions_output/fairness_metrics_cot.csv")

# # Define labels for each dataset
# labels_before = ["Man", "Nonbinary", "Woman"]  # For the first dataset (Before Bias Mitigation)
# labels_counter = ["Man", "Woman"]  # For the second dataset (After Counterfactual Augmentation)
# labels_cot = ["Man", "NonBinary", "Woman"]  # For the third dataset (After CoT Prompting)

# # Function to plot graphs
# # Function to plot graphs
# def plot_fairness_metrics(df, title_prefix, save_prefix, labels):
#     """
#     Generates and saves fairness metric graphs for a given dataset.
#     """
#     plt.figure(figsize=(12, 4))

#     # Demographic Parity
#     plt.subplot(1, 3, 1)
#     sns.barplot(x=labels, y=df["Demographic Parity"], palette="coolwarm")
#     plt.title(f"{title_prefix} - Demographic Parity")
#     plt.ylabel("Score")
#     plt.xlabel("Gender")

#     # Equalized Odds (False Positive Rate & False Negative Rate)
#     plt.subplot(1, 3, 2)
#     sns.barplot(x=labels, y=df["False Positive Rate (Equalized Odds)"], label="FPR", color="lightcoral")
#     sns.barplot(x=labels, y=df["False Negative Rate (Equalized Odds)"], label="FNR", color="steelblue", alpha=0.7)
#     plt.title(f"{title_prefix} - Equalized Odds")
#     plt.ylabel("Score")
#     plt.xlabel("Gender")
#     plt.legend()

#     # Predictive Rate Parity (Qualified & Unqualified Rates)
#     plt.subplot(1, 3, 3)
#     sns.barplot(x=labels, y=df["Qualified Rate (Predictive Rate Parity)"], label="Qualified", color="seagreen")
#     sns.barplot(x=labels, y=df["Unqualified Rate (Predictive Rate Parity)"], label="Unqualified", color="goldenrod", alpha=0.7)
#     plt.title(f"{title_prefix} - Predictive Rate Parity")
#     plt.ylabel("Score")
#     plt.xlabel("Gender")
#     plt.legend()

#     # Adjust title for the second graph to avoid overlap
#     if save_prefix == "counter":
#         plt.subplot(1, 3, 1)  # Title for the first graph (Demographic Parity)
#         plt.title(f"{title_prefix}\n(Note: Male swapped with Woman)", fontsize=10)

#     plt.tight_layout()
#     plt.savefig(f"graphs/{save_prefix}_fairness_metrics.png")
#     plt.show()

# # Generate graphs for all three datasets
# plot_fairness_metrics(fairness_metrics, "Before Bias Mitigation", "before", labels_before)
# plot_fairness_metrics(fairness_metrics_counter, "After Counterfactual Augmentation", "counter", labels_counter)
# plot_fairness_metrics(fairness_metrics_cot, "After CoT Prompting", "cot", labels_cot)

# print("Graphs generated and saved in 'graphs/' folder.")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output directory for graphs if it doesn't exist
os.makedirs("graphs", exist_ok=True)

# Load the datasets (With correct column names)
# Make sure your CSV files are in a folder named 'predictions_output'
try:
    fairness_metrics = pd.read_csv("predictions_output/fairness_metrics_results.csv")
    fairness_metrics_counter = pd.read_csv("predictions_output/fairness_metrics_results_counterfactual.csv")
    fairness_metrics_cot = pd.read_csv("predictions_output/fairness_metrics_cot.csv")
    print("Successfully loaded CSV files.")

except FileNotFoundError:
    print("Warning: Could not find CSV files in 'predictions_output/'.")
    print("Using dummy data to generate plots. Please verify file paths.")
    # Create dummy dataframes if files are not found
    fairness_metrics = pd.DataFrame({
        "Demographic Parity": [0.7, 0.5, 0.71],
        "False Positive Rate (Equalized Odds)": [0.42, 0.5, 0.42],
        "False Negative Rate (Equalized Odds)": [0.08, 0.1, 0.12],
        "Qualified Rate (Predictive Rate Parity)": [0.6, 0.4, 0.7],
        "Unqualified Rate (Predictive Rate Parity)": [0.3, 0.2, 0.3]
    })
    fairness_metrics_counter = pd.DataFrame({
        "Demographic Parity": [0.22, 0.19],
        "False Positive Rate (Equalized Odds)": [0.11, 0.12],
        "False Negative Rate (Equalized Odds)": [0.11, 0.08],
        "Qualified Rate (Predictive Rate Parity)": [0.33, 0.23],
        "Unqualified Rate (Predictive Rate Parity)": [0.17, 0.14]
    })
    fairness_metrics_cot = pd.DataFrame({
        "Demographic Parity": [0.01, 0.0, 0.01],
        "False Positive Rate (Equalized Odds)": [0.0, 0.0, 0.0],
        "False Negative Rate (Equalized Odds)": [0.45, 0.46, 0.45],
        "Qualified Rate (Predictive Rate Parity)": [0.0, 0.0, 0.0],
        "Unqualified Rate (Predictive Rate Parity)": [0.0, 0.0, 0.0]
    })

# Define labels for each dataset
labels_before = ["Man", "Nonbinary", "Woman"]
labels_counter = ["Man", "Woman"]
labels_cot = ["Man", "NonBinary", "Woman"]

# --- MODIFIED PLOTTING FUNCTION ---
def plot_fairness_metrics(df, title_prefix, save_prefix, labels):
    """
    Generates and saves fairness metric graphs for a given dataset.
    Titles for the 'counter' set are modified to avoid overlap.
    """
    plt.figure(figsize=(12, 4))

    # --- Subplot 1: Demographic Parity ---
    plt.subplot(1, 3, 1)
    sns.barplot(x=labels, y=df["Demographic Parity"], palette="coolwarm")
    # Special title handling for counterfactual
    if save_prefix == "counter":
        plt.title(f"{title_prefix}\n(Note: Male swapped with Woman)", fontsize=10)
    else:
        plt.title(f"{title_prefix} - Demographic Parity")
    plt.ylabel("Score")
    plt.xlabel("Gender")

    # --- Subplot 2: Equalized Odds ---
    plt.subplot(1, 3, 2)
    sns.barplot(x=labels, y=df["False Positive Rate (Equalized Odds)"], label="FPR", color="lightcoral")
    sns.barplot(x=labels, y=df["False Negative Rate (Equalized Odds)"], label="FNR", color="steelblue", alpha=0.7)
    # Special title handling for counterfactual
    if save_prefix == "counter":
        plt.title("Equalized Odds") # Simplified title
    else:
        plt.title(f"{title_prefix} - Equalized Odds")
    plt.ylabel("Score")
    plt.xlabel("Gender")
    plt.legend()

    # --- Subplot 3: Predictive Rate Parity ---
    plt.subplot(1, 3, 3)
    sns.barplot(x=labels, y=df["Qualified Rate (Predictive Rate Parity)"], label="Qualified", color="seagreen")
    sns.barplot(x=labels, y=df["Unqualified Rate (Predictive Rate Parity)"], label="Unqualified", color="goldenrod", alpha=0.7)
    # Special title handling for counterfactual
    if save_prefix == "counter":
        plt.title("Predictive Rate Parity") # Simplified title
    else:
        plt.title(f"{title_prefix} - Predictive Rate Parity")
    plt.ylabel("Score")
    plt.xlabel("Gender")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"graphs/{save_prefix}_fairness_metrics.png")
    # plt.show() # Removed for server environment

# Generate graphs for all three datasets
# Ensure we only try to plot data for the labels we have
plot_fairness_metrics(fairness_metrics.head(len(labels_before)), "Before Bias Mitigation", "before", labels_before)
plot_fairness_metrics(fairness_metrics_counter.head(len(labels_counter)), "After Counterfactual Augmentation", "counter", labels_counter)
plot_fairness_metrics(fairness_metrics_cot.head(len(labels_cot)), "After CoT Prompting", "cot", labels_cot)

print("Graphs generated and saved in 'graphs/' folder.")