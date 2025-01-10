import torch

combined_dict = {}
logdir = "/users/psi/yjia/espnet/espnet/egs2/calibration/asr1/exp/asr_train_asr_conformer_raw_en_bpe5000/decode_asr_bs20_asr_model_pretrained/test_lib360/logdir"
# logdir = "exp/asr_train_asr_conformer_raw_en_bpe5000/decode_asr_asr_model_pretrained/org/dev_lib360/logdir"
job_num = 1
for i in range(job_num):
    scores_path = logdir + "/output." + str(i+1) + "/1best_recog/scores_list.pth"
    scores_dict = torch.load(scores_path)
    combined_dict.update(scores_dict)
    print(str(i+1) + '/' + str(job_num))

torch.save(combined_dict, logdir + '/combined_scores.pth')