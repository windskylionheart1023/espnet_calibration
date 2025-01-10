import torch

def weighted_decoder_output_ys(
        decoder_out,
        ys_out_pad,
        padding_id = -1,
):
    predicted_classes = torch.argmax(decoder_out, dim=-1)  # Shape: (batch, seq_len)
    non_padding_mask = ys_out_pad != padding_id  # Shape: (batch, seq_len)

    correct_mask = (predicted_classes == ys_out_pad) & non_padding_mask  # Correct predictions
    incorrect_mask = (~correct_mask) & non_padding_mask  # Incorrect predictions
    if torch.all(correct_mask) or torch.all(~correct_mask):
        return decoder_out, ys_out_pad
    else:
        correct_ys_out_pad = ys_out_pad[correct_mask]  # Correct predictions
        correct_decoder_out = decoder_out[correct_mask] # Their corresponding targets

        incorrect_ys_out_pad = ys_out_pad[incorrect_mask]  # Incorrect predictions
        incorrect_decoder_out = decoder_out[incorrect_mask]  # Their corresponding targets

        return correct_decoder_out, correct_ys_out_pad, incorrect_decoder_out, incorrect_ys_out_pad
