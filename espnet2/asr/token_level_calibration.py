from AdaptiveBinning import AdaptiveBinning
from espnet2.text.token_id_converter import TokenIDConverter
import matplotlib.pyplot as plt
import json
import re

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

def parse_file_to_dict(filepath):
    result = {}
    id_pattern = re.compile(r"^\d+-\d+-\d+")
    
    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            
            match = id_pattern.match(line)
            if match:
                id_str = match.group(0)  # Extract ID
                content = line[len(id_str):].strip().split()  # Split remaining content into list
                
                # Convert content to int if possible
                content = [int(x) if x.isdigit() else x for x in content]
                result[id_str] = content
    
    return result

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
    # scores_path = logdir + f"/output.1/{n}best_recog/scores_list_before_pre_beam.json"
    scores_path = logdir + f"/output.1/{n}best_recog/scores_list.json"
    with open(scores_path, "r") as file:
        scores_dict = json.load(file)

    # Step 3: Initialize the token ID converter
    tokenidconvertor = TokenIDConverter(token_path)

    hyp_path = logdir + f"/output.1/{n}best_recog/token_int"
    hyp_int_dict = parse_file_to_dict(hyp_path)

    counter_utt = 0
    counter_token = 0
    infer_pair_list = []

    for pair in scores_dict:
        score_id = pair[0]
        # label_id = '-'.join(score_id.split('-')[-3:])
        label_id = f"{score_id}-{score_id.split('-')[-2]}-{score_id.split('-')[-1]}"
        label_id = f"{score_id.split('-')[0]}-{score_id.split('-')[1]}-{score_id}"
        label = label_dict[label_id]['REF']
        score = pair[1]
        hyp = label_dict[label_id]['HYP']

        hyp_int = hyp_int_dict[score_id]

        # Filter out tokens with '*' in the hypothesis and label
        del_list = [False if '*' in t else True for t in hyp]
        hyp = [t for t, include in zip(hyp, del_list) if include]
        label = [t for t, include in zip(label, del_list) if include]

        if len(hyp) != len(hyp_int):
            counter_utt += 1
            print(f'{n}best label_id:{score_id} hyp and hyp_int lengths do not match.')
            continue
        
        # Further clean up the label and score based on '*' tokens
        ins_del_list = [False if '*' in t else True for t in label]
        label = [t for t, include in zip(label, ins_del_list) if include]
        score = [t for t, include in zip(score, ins_del_list) if include]
        hyp = [t for t, include in zip(hyp, ins_del_list) if include]
        hyp_int = [t for t, include in zip(hyp_int, ins_del_list) if include]

        label = [l.upper() for l in label]
        hyp = [h.upper() for h in hyp]

        # Ensure that the score and label lengths are equal
        if len(score) != len(label):
             print(f'{n}best label_id:{score_id} Score and label lengths do not match.')
             continue

        # Convert tokens to IDs and prepare them as tensors
        label = tokenidconvertor.tokens2ids(label)
        hyp = tokenidconvertor.tokens2ids(hyp)
        
        for i in range(len(hyp)):
            if hyp_int[i] == hyp[i]:
                conf = score[i][str(hyp[i])]
                corr = 1 if hyp[i] == label[i] else 0
            else:
                counter_token += 1
                print('tokenized token does not match original token')
            infer_pair_list.append((conf, corr))
        # Append the processed labels and scores to the lists
        # all_score += score
        # all_label += label
        # all_hyp += hyp
    print(f'skiped token number: {counter_token}')
    print(f'skiped utt number: {counter_utt}')

    return infer_pair_list

# Example usage
decode_path = "/users/psi/yjia/espnet_beta/espnet/egs2/calibration/asr1/exp/asr_train_asr_conformer_raw_en_bpe5000/decode_asr_bs20_asr_model_Focal_gamma1/test_lib360"
logdir = decode_path + "/logdir"

token_path = "/users/psi/yjia/espnet/espnet/egs2/calibration/asr1/data/en_token_list/bpe_unigram5000/tokens.txt"

n_infer_pair_list = []
nbest = 1
for n in range(1, nbest+1):
    label_file = decode_path + f"/score_ter/{n}best_recog/result.txt"
    n_infer_pair_list = process_data(label_file, logdir, token_path, n)
    n_infer_pair_list += n_infer_pair_list

print("The number of tokens for ECE:")
print(len(n_infer_pair_list))

# Call AdaptiveBinning.
AECE, AMCE, confidence, accuracy, cof_min, cof_max = AdaptiveBinning(n_infer_pair_list, True)
print('ECE based on adaptive binning: {}'.format(round(100 * AECE, 2)))
print('MCE based on adaptive binning: {}'.format(round(100 * AMCE, 2)))

plt.savefig("/users/psi/yjia/espnet_beta/espnet/espnet2/asr/reliability_diagram_token.png")
