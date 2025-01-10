from espnet2.asr.ece_kde import get_bandwidth, get_ece_kde
from espnet2.text.token_id_converter import TokenIDConverter
import json
import torch
import argparse

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
    
    # Step 1: Load label file into dictionary
    label_dict = process_file_to_dict(label_file)

    # Step 2: Load scores from log directory
    # scores_path = logdir + f"/output.1/{n}best_recog/scores_list.json"
    scores_path = logdir + f"/output.1/{n}best_recog/scores_list_before_pre_beam.json"
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
            #  print(n)
            #  print(label_id)
             print("Score and label lengths do not match.")
             continue

        # Convert tokens to IDs and prepare them as tensors
        label = tokenidconvertor.tokens2ids(label)

        # Append the processed labels and scores to the lists
        all_score.append(score)
        all_label.append(label)

    return all_label, all_score

def convert_to_tensor(data, num_classes):
    # Number of samples
    num_samples = len(data)
    
    # Initialize the tensor with zeros
    tensor = torch.zeros((num_samples, num_classes))
    
    # Populate the tensor
    for i, sample in enumerate(data):
        for class_index, value in sample.items():
            tensor[i, int(class_index)] = value
    
    return tensor

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process decode path for calibration")
    parser.add_argument("decode_path", type=str, help="Path to the decode directory")
    args = parser.parse_args()

    # Use the decode_path from the command-line argument
    decode_path = args.decode_path
    logdir = decode_path + "/logdir"

    token_path = "/users/psi/yjia/espnet/espnet/egs2/calibration/asr1/data/en_token_list/bpe_unigram5000/tokens.txt"
    label_file = decode_path + f"/score_ter/1best_recog/result.txt"

    labels, logits = process_data(label_file, logdir, token_path, 1)

    batch_size = 100
    print(f'batch_size: {batch_size}')
    ece_kde_list = []
    for i in range(0, len(labels), batch_size):
        print(f'Processing batch {i} to {i + batch_size}, total {len(labels)}')
        batch_labels = labels[i:i + batch_size]
        batch_logits = logits[i:i + batch_size]

        batch_labels = sum(batch_labels, [])
        batch_logits = sum(batch_logits, [])

        f = convert_to_tensor(batch_logits, 5000)
        # Replace zero values with epsilon
        epsilon = 1e-10
        f = torch.where(f == 0, torch.tensor(epsilon, device=f.device), f)
        remaining_sums = f.sum(dim=1, keepdim=True)
        f = f / remaining_sums

        y = torch.tensor(batch_labels)

        bandwidth = get_bandwidth(f, device=f.device)
        ece_kde = get_ece_kde(f, y, bandwidth=bandwidth, p=1, mc_type='canonical', device=f.device)
        ece_kde_list.append(ece_kde.item())
        print(f'ECE KDE: {ece_kde.item()}')
    print(sum((ece_kde_list)) / len(ece_kde_list))

# /users/psi/yjia/espnet_beta/espnet/egs2/calibration/asr1/exp/asr_train_asr_conformer_raw_en_bpe5000/decode_asr_bs20_asr_model_Focal/test_lib360
# Processing batch 0 to 100, total 2528
# ECE KDE: 0.5123196840286255
# Processing batch 100 to 200, total 2528
# ECE KDE: 0.5059456825256348
# Processing batch 200 to 300, total 2528
# ECE KDE: 0.501874566078186
# Processing batch 300 to 400, total 2528
# ECE KDE: 0.49350476264953613
# Processing batch 400 to 500, total 2528
# ECE KDE: 0.5569366812705994
# 0.5141

# /users/psi/yjia/espnet_beta/espnet/egs2/calibration/asr1/exp/asr_train_asr_conformer_raw_en_bpe5000/decode_asr_bs20_asr_model_KDE_lambda10/test_lib360
# Processing batch 0 to 100, total 2528
# ECE KDE: 0.4999223053455353
# Processing batch 100 to 200, total 2528
# ECE KDE: 0.4928508400917053
# Processing batch 200 to 300, total 2528
# ECE KDE: 0.48086535930633545
# Processing batch 300 to 400, total 2528
# ECE KDE: 0.4732924997806549
# Processing batch 400 to 500, total 2528
# ECE KDE: 0.5308283567428589
# 0.4956

# /users/psi/yjia/espnet_beta/espnet/egs2/calibration/asr1/exp/asr_train_asr_conformer_raw_en_bpe5000/decode_asr_bs20_asr_model_pretrain40e/test_lib360
# Processing batch 0 to 100, total 2514
# ECE KDE: 0.5292972326278687
# Processing batch 100 to 200, total 2514
# ECE KDE: 0.5033809542655945
# Processing batch 200 to 300, total 2514
# ECE KDE: 0.49721723794937134
# Processing batch 300 to 400, total 2514
# ECE KDE: 0.5355600714683533
# Processing batch 400 to 500, total 2514
# ECE KDE: 0.550178050994873
# 0.5231

# /users/psi/yjia/espnet_beta/espnet/egs2/calibration/asr1/exp/asr_train_asr_conformer_raw_en_bpe5000/decode_asr_bs20_asr_model_MMCE/test_lib360
# Processing batch 0 to 100, total 2528
# ECE KDE: 0.49847593903541565
# Processing batch 100 to 200, total 2528
# ECE KDE: 0.4896165430545807
# Processing batch 200 to 300, total 2528
# ECE KDE: 0.4662308096885681
# Processing batch 300 to 400, total 2528
# ECE KDE: 0.47650396823883057
# Processing batch 400 to 500, total 2528
# ECE KDE: 0.5345572829246521
# 0.4931

# /users/psi/yjia/espnet_beta/espnet/egs2/calibration/asr1/exp/asr_train_asr_conformer_raw_en_bpe5000/decode_asr_bs20_asr_model_pretrain40e_without_TS/test_lib360
# Processing batch 0 to 100, total 2528
# ECE KDE: 0.490438848733902
# Processing batch 100 to 200, total 2528
# ECE KDE: 0.4835440516471863
# Processing batch 200 to 300, total 2528
# ECE KDE: 0.4852317273616791
# Processing batch 300 to 400, total 2528
# ECE KDE: 0.46781134605407715
# Processing batch 400 to 500, total 2528
# ECE KDE: 0.5278978943824768
# 0.4910

# /users/psi/yjia/espnet_beta/espnet/egs2/calibration/asr1/exp/asr_train_asr_conformer_raw_en_bpe5000/decode_asr_bs20_asr_model_pretrain80e/test_lib360
# Processing batch 0 to 100, total 2528
# ECE KDE: 0.5056747198104858
# Processing batch 100 to 200, total 2528
# ECE KDE: 0.48672327399253845
# Processing batch 200 to 300, total 2528
# ECE KDE: 0.4602118730545044
# Processing batch 300 to 400, total 2528
# ECE KDE: 0.46685948967933655
# Processing batch 400 to 500, total 2528
# ECE KDE: 0.5281475186347961

# ----
# /users/psi/yjia/espnet_beta/espnet/egs2/calibration/asr1/exp/asr_train_asr_conformer_raw_en_bpe5000/decode_asr_bs20_asr_model_Focal/small_test_lib360
# 51.90

# /users/psi/yjia/espnet_beta/espnet/egs2/calibration/asr1/exp/asr_train_asr_conformer_raw_en_bpe5000/decode_asr_bs20_asr_model_Focal40e/small_test_lib360
# 57.23

# /users/psi/yjia/espnet_beta/espnet/egs2/calibration/asr1/exp/asr_train_asr_conformer_raw_en_bpe5000/decode_asr_bs20_asr_model_MMCE/small_test_lib360
# 52.70

# /users/psi/yjia/espnet_beta/espnet/egs2/calibration/asr1/exp/asr_train_asr_conformer_raw_en_bpe5000/decode_asr_bs20_asr_model_KDE_lambda10/small_test_lib360
# 56.59