import torch

bs1_scores_path = "/users/psi/yjia/espnet_beta/espnet/egs2/calibration/asr1/exp/asr_train_asr_conformer_raw_en_bpe5000/decode_asr_bs1_asr_model_10epoch/org/dev_lib360/logdir/output.1/1best_recog/scores_list.pth"
bs2_scores_path = "/users/psi/yjia/espnet_beta/espnet/egs2/calibration/asr1/exp/asr_train_asr_conformer_raw_en_bpe5000/decode_asr_bs2_asr_model_10epoch/org/dev_lib360/logdir/output.1/1best_recog/scores_list.pth"
bs20_scores_path = "/users/psi/yjia/espnet_beta/espnet/egs2/calibration/asr1/exp/asr_train_asr_conformer_raw_en_bpe5000/decode_asr_asr_model_pretrained/org/dev_lib360/logdir/output.1/1best_recog/scores_list.pth"

bs1_scores = torch.load(bs1_scores_path)
bs2_scores = torch.load(bs2_scores_path)
bs20_scores = torch.load(bs20_scores_path)

id = list(bs1_scores.keys())[0]

part_bs1_scores = bs1_scores[id][0][0]
part_bs2_scores = bs2_scores[id][0][0]
part_bs20_scores = bs20_scores[id][0][0]

pass