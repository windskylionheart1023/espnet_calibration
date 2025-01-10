import torch
from torch import nn, optim
from espnet2.text.token_id_converter import TokenIDConverter
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
             print(n)
             print(label_id)
            #  print("Score and label lengths do not match.")
             continue

        # Convert tokens to IDs and prepare them as tensors
        label = tokenidconvertor.tokens2ids(label)

        # Append the processed labels and scores to the lists
        all_score += score
        all_label += label

    return all_label, all_score

# Example usage
decode_path = "/users/psi/yjia/espnet_beta/espnet/egs2/calibration/asr1/exp/asr_train_asr_conformer_raw_en_bpe5000/decode_asr_bs20_asr_model_pretrain40e/org/dev_lib360"
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

labels = torch.tensor(labels).cuda()

# Number of dimensions for each tensor (e.g., 5000)
tensor_dim = 5000

# Convert each dictionary into a tensor
tensors = []
for sparse_dict in logits:
    tensor = torch.zeros(tensor_dim)
    for key, value in sparse_dict.items():
        tensor[int(key)] = value
    tensors.append(tensor)

# Stack all tensors into a batch tensor
batch_tensor = torch.stack(tensors)

# print(batch_tensor.shape)  # (num_dicts, tensor_dim)
# print(batch_tensor)

logits = batch_tensor.cuda()

temperature = torch.ones(1).cuda() * 1.5
temperature.requires_grad_(True)
nll_criterion = nn.CrossEntropyLoss().cuda()

def temperature_scale(temperature, logits):
    """
    Perform temperature scaling on logits
    """
    # Expand temperature only if needed, without making it a non-leaf tensor
    return logits / temperature  # Broadcasting will happen automatically

# Next: optimize the temperature w.r.t. NLL
optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)

def eval():
    optimizer.zero_grad()
    loss = nll_criterion(temperature_scale(temperature, logits), labels)
    loss.backward()
    return loss
optimizer.step(eval)

print(temperature)
# bs=1
# tensor([1.1853], device='cuda:0', requires_grad=True)

# decode_asr_bs20_asr_model_pretrained_without_TS
# tensor([1.1450], device='cuda:0', requires_grad=True)

# espnet_beta/espnet/egs2/calibration/asr1/exp/asr_train_asr_conformer_raw_en_bpe5000/decode_asr_bs20_asr_model_pretrain40e/org/dev_lib360
# tensor([1.2550], device='cuda:0', requires_grad=True)

# I need to store all weighted_scores at each step
# The orignal implementation do not keep these values

# identify these values and pass them out, store all in a variable then store it in a file?