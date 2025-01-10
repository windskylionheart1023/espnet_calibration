import json
import os
import numpy as np
import re

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

path = '/users/psi/yjia/espnet_beta/espnet/egs2/calibration/asr1/exp/asr_train_asr_conformer_raw_en_bpe5000/decode_asr_bs20_asr_model_Focal_gamma1/test_lib360/'
prob_per_token_path = os.path.join(path, 'logdir/output.1/1best_recog/scores_list.json')
# prob_per_token_path = os.path.join(path, 'scores_list.json')
text_path = os.path.join(path, 'logdir/output.1/1best_recog/text')
token_path = os.path.join(path, 'logdir/output.1/1best_recog/token')
token_int_path = os.path.join(path, 'logdir/output.1/1best_recog/token_int')

# Example usage
text_dict = parse_file_to_dict(text_path)
token_dict = parse_file_to_dict(token_path)
token_int_dict = parse_file_to_dict(token_int_path)

with open(prob_per_token_path, "r") as file:
    prob_per_token_dict = json.load(file)

word_correctness_path = os.path.join(path, 'score_wer/result.txt')

word_correctness_dict = process_file_to_dict(word_correctness_path)

conf_corr_list = []

for pair in prob_per_token_dict:
    id = pair[0]
    prob_dict = pair[1]
    word_boundary = [i for i, token in enumerate(token_dict[id]) if token.startswith('▁')]
    word_boundary.append(len(token_dict[id]))
    if word_boundary[0] != 0:
        word_boundary.insert(0, 0)
    confidence_list = [prob_dict[index][str(i)] for index, i in enumerate(token_int_dict[id])]
    word_confidence_block = [confidence_list[word_boundary[i]:word_boundary[i+1]] for i in range(len(word_boundary)-1)]
    word_confidence = [np.min(block) for block in word_confidence_block]
    
    label_id = f"{id.split('-')[0]}-{id.split('-')[1]}-{id}"
    label = word_correctness_dict[label_id]['REF']
    hyp = word_correctness_dict[label_id]['HYP']

    # Filter out tokens with '*' in the hypothesis and label
    del_list = [False if '*' in t else True for t in hyp]
    hyp = [t for t, include in zip(hyp, del_list) if include]
    label = [t for t, include in zip(label, del_list) if include]

    assert len(hyp) == len(word_confidence), f"Lengths do not match for {id}" 
    
    # Further clean up the label and score based on '*' tokens
    ins_del_list = [False if '*' in t else True for t in label]
    hyp = [t for t, include in zip(hyp, ins_del_list) if include]
    label = [t for t, include in zip(label, ins_del_list) if include]
    word_confidence = [t for t, include in zip(word_confidence, ins_del_list) if include]
    
    label = [l.upper() for l in label]
    hyp = [h.upper() for h in hyp]

    for i in range(len(label)):
        conf = word_confidence[i]
        corr = 1 if label[i] == hyp[i] else 0
        conf_corr_list.append((conf, corr))

from AdaptiveBinning import AdaptiveBinning
import matplotlib.pyplot as plt

AECE, AMCE, confidence, accuracy, cof_min, cof_max = AdaptiveBinning(conf_corr_list, True)
print('ECE based on adaptive binning: {}'.format(round(100 * AECE, 2)))
print('MCE based on adaptive binning: {}'.format(round(100 * AMCE, 2)))

plt.savefig("/users/psi/yjia/espnet_beta/espnet/espnet2/asr/reliability_diagram_word.png")