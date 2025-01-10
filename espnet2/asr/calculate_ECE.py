from espnet2.asr.ece_kde import get_ece_kde, get_bandwidth
import torch
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
                "Scores": lines[i+1].rstrip(),  # Keep spaces, remove trailing newline
                "REF": lines[i+2].rstrip(),
                "HYP": lines[i+3].rstrip(),
                "Eval": lines[i+4].rstrip()
            }

            values['REF'] = values['REF'][6:].split()
            values['HYP'] = values['HYP'][6:].split()

            data_dict[item_id]=values

    return data_dict

# Example usage:
label_file = "/users/psi/yjia/espnet_beta/espnet/egs2/calibration/asr1/exp/asr_train_asr_conformer_raw_en_bpe5000/decode_asr_bs20_asr_model_Focal80e/test_lib360/score_ter/1best_recog/result.txt"
label_dict = process_file_to_dict(label_file)

logdir = "/users/psi/yjia/espnet_beta/espnet/egs2/calibration/asr1/exp/asr_train_asr_conformer_raw_en_bpe5000/decode_asr_bs20_asr_model_Focal80e/test_lib360/logdir"
socres_path = logdir + "/.pth"
socres_dict = torch.load(socres_path)

token_path = "/users/psi/yjia/espnet_beta/espnet/egs2/calibration/asr1/data/en_token_list/bpe_unigram5000/tokens.txt"
tokenidconvertor = TokenIDConverter(token_path)

ece_loss_list = []

for label_id in list(label_dict.keys()):
    score_id = '-'.join(label_id.split('-')[-3:])
    label = label_dict[label_id]['REF']
    score = socres_dict[score_id]
    hyp = label_dict[label_id]['HYP']

    del_list = [False if '*' in t else True for t in hyp]
    hyp = [t for t, include in zip(hyp, del_list) if include]
    label = [t for t, include in zip(label, del_list) if include]
    
    ins_del_list = [False if '*' in t else True for t in label]
    label = [t for t, include in zip(label, ins_del_list) if include]
    score = [t for t, include in zip(score, ins_del_list) if include]

    score = [s[0].unsqueeze(0) for s in score]
    label = [l.upper() for l in label]

    assert len(score)==len(label), "not equal length"

    label = tokenidconvertor.tokens2ids(label)
    label = [torch.tensor(l).unsqueeze(0) for l in label]


    f = torch.cat(score).cuda().softmax(1)
    y = torch.cat(label).cuda()

    bandwidth = get_bandwidth(f, f.device)
    ece_one_utt = get_ece_kde(f, y, bandwidth, 1, "canonical", f.device)
    ece_loss_list.append(ece_one_utt)

average = torch.mean(torch.stack(ece_loss_list))
print(average)