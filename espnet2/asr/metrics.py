from AdaptiveBinning import AdaptiveBinning
from espnet2.text.token_id_converter import TokenIDConverter
import matplotlib.pyplot as plt
import json

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
                "Scores": lines[i+1].rstrip(),  # Keep spaces, remove trailing newline
                "REF": lines[i+2].rstrip(),
                "HYP": lines[i+3].rstrip(),
                "Eval": lines[i+4].rstrip()
            }

            values['REF'] = values['REF'][6:].split()
            values['HYP'] = values['HYP'][6:].split()

            data_dict[item_id]=values

    return data_dict

def process_data(label_file, logdir, token_path, n):
    """
    Processes the label and score data and returns the concatenated labels and logits.
    
    Args:
    - label_file (str): Path to the label file to be processed.
    - logdir (str): Path to the log directory containing the score files.
    - token_path (str): Path to the token file used for token ID conversion.
    
    Returns:
    - labels (torch.Tensor): Concatenated tensor of label IDs.
    - logits (torch.Tensor): Concatenated tensor of logits (predicted scores).
    """
    
    # Step 1: Load label file into dictionary
    label_dict = process_file_to_dict(label_file)

    # Step 2: Load scores from log directory
    scores_path = logdir + f"/output.1/{n}best_recog/scores_list.json"
    with open(scores_path, "r") as file:
        scores_dict = json.load(file)

    # Step 3: Initialize the token ID converter
    tokenidconvertor = TokenIDConverter(token_path)

    # Initialize lists for labels and scores
    all_label = []
    all_score = []

    # Step 4: Process each label and corresponding score
    for pair in scores_dict:
        score_id = pair[0]
        # label_id = '-'.join(score_id.split('-')[-3:])
        label_id = f"{score_id}-{score_id.split('-')[-2]}-{score_id.split('-')[-1]}"
        label_id = f"{score_id.split('-')[0]}-{score_id.split('-')[1]}-{score_id}"
        label = label_dict[label_id]['REF']
        score = pair[1]
        hyp = label_dict[label_id]['HYP']

        # Filter out tokens with '*' in the hypothesis and label
        del_list = [False if '*' in t else True for t in hyp]
        hyp = [t for t, include in zip(hyp, del_list) if include]
        label = [t for t, include in zip(label, del_list) if include]
        
        # Further clean up the label and score based on '*' tokens
        ins_del_list = [False if '*' in t else True for t in label]
        label = [t for t, include in zip(label, ins_del_list) if include]
        score = [t for t, include in zip(score, ins_del_list) if include]

        label = [l.upper() for l in label]

        # Ensure that the score and label lengths are equal
        if len(score) != len(label):
             print("Score and label lengths do not match.")
             continue

        # Convert tokens to IDs and prepare them as tensors
        label = tokenidconvertor.tokens2ids(label)

        # Append the processed labels and scores to the lists
        all_score += score
        all_label += label

    return all_label, all_score

# Example usage
decode_path = "/users/psi/yjia/espnet/espnet/egs2/calibration/asr1/exp/asr_train_asr_conformer_KDE_raw_en_bpe5000/decode_asr_bs20_asr_model_valid.acc.ave_10best/small_test_lib360"
logdir = decode_path + "/logdir"

token_path = "/users/psi/yjia/espnet/espnet/egs2/calibration/asr1/data/en_token_list/bpe_unigram5000/tokens.txt"

labels = []
logits = []
nbest = 1
for n in range(1, nbest+1):
    label_file = decode_path + f"/score_ter/{n}best_recog/result.txt"
    nbest_labels, nbest_logits = process_data(label_file, logdir, token_path, n)
    labels += nbest_labels
    logits += nbest_logits

print(len(labels))

prediction = [max(data, key=data.get) for data in logits]

infer_results = []

for i in range(len(prediction)):
	correctness = (labels[i] == int(prediction[i]))
	infer_results.append([logits[i][prediction[i]], correctness])

# Call AdaptiveBinning.
AECE, AMCE, confidence, accuracy, cof_min, cof_max = AdaptiveBinning(infer_results, True)

print('ECE based on adaptive binning: {}'.format(AECE))
print('MCE based on adaptive binning: {}'.format(AMCE))

plt.savefig("/users/psi/yjia/espnet/espnet/espnet2/asr/reliability_diagram.png")

# from espnet2.asr.ece_kde import *
# bandwidth = get_bandwidth(probability, device='cuda')
# ece_kde = get_ece_kde(probability, labels, bandwidth=bandwidth, p=1, mc_type='canonical', device='cuda')
# print(ece_kde)
