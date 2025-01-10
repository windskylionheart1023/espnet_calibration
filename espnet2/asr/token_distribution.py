import json
import matplotlib.pyplot as plt
from espnet2.text.token_id_converter import TokenIDConverter

def process_file_to_dict(input_path):
    with open(input_path, 'r') as file:
        lines = file.readlines()

    # Step 1: Find the first occurrence of "Speaker sentences" and delete everything above it
    start_idx = next(i for i, line in enumerate(lines) if "Speaker sentences" in line)
    lines = lines[start_idx:]

    # Step 2: Process each occurrence of "id: (" and take the next 5 lines
    data_dict = {}
    for i, line in enumerate(lines):
        if line.startswith("id: ("):
            # Extract the id without parentheses
            item_id = line.split()[1].strip('()')
            # Collect the next four lines after the "id" line, keep spaces using rstrip()
            values = {
                "REF": lines[i+2].rstrip()[6:].split()
            }

            data_dict[item_id] = values

    return data_dict

def process_labels(label_file, logdir, token_path, n):
    """
    Extracts and processes labels from the input files.

    Args:
    - label_file (str): Path to the label file to be processed.
    - logdir (str): Path to the log directory containing the score files.
    - token_path (str): Path to the token file used for token ID conversion.

    Returns:
    - labels (list): List of token IDs representing labels.
    """
    # Step 1: Load label file into dictionary
    label_dict = process_file_to_dict(label_file)

    # Step 2: Load scores from log directory
    scores_path = logdir + f"/output.1/{n}best_recog/scores_list.json"
    with open(scores_path, "r") as file:
        scores_dict = json.load(file)

    # Step 3: Initialize the token ID converter
    tokenidconvertor = TokenIDConverter(token_path)

    # Initialize list for labels
    all_labels = []

    # Step 4: Process each label and corresponding score
    for pair in scores_dict:
        score_id = pair[0]
        label_id = f"{score_id.split('-')[0]}-{score_id.split('-')[1]}-{score_id}"
        label = label_dict[label_id]['REF']

        # Convert tokens to uppercase and filter out tokens with '*'
        label = [t.upper() for t in label if '*' not in t]

        # Convert tokens to IDs
        label = tokenidconvertor.tokens2ids(label)

        # Append the processed labels to the list
        all_labels += label

    return all_labels

# Example usage
decode_path = "/users/psi/yjia/espnet_beta/espnet/egs2/calibration/asr1/exp/asr_train_asr_conformer_raw_en_bpe5000/decode_asr_bs20_asr_model_MMCE/test_lib360"
logdir = decode_path + "/logdir"

token_path = "/users/psi/yjia/espnet/espnet/egs2/calibration/asr1/data/en_token_list/bpe_unigram5000/tokens.txt"

labels = []
nbest = 1
for n in range(1, nbest+1):
    label_file = decode_path + f"/score_ter/{n}best_recog/result.txt"
    nbest_labels = process_labels(label_file, logdir, token_path, n)
    labels += nbest_labels

# Plotting the label distribution
plt.figure(figsize=(10, 6))
plt.hist(labels, bins=range(min(labels), max(labels) + 1), edgecolor="black", align='left')
plt.title("Distribution of Labels")
plt.xlabel("Label IDs")
plt.ylabel("Frequency")
# plt.show()
plt.savefig("/users/psi/yjia/espnet_beta/espnet/espnet2/asr/class_distribution.png")
