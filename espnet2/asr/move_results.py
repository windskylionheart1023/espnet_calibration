import os
import shutil

# Define the result_path
result_path = "/users/psi/yjia/espnet_beta/espnet/egs2/calibration/asr1/exp/asr_train_asr_conformer_raw_en_bpe5000/decode_asr_bs20_asr_model_Focal/test_lib360/score_ter"

# Define the variable n
n = 3  # Replace this with your desired value

# Create the folder "{n}best_recog" in result_path
new_folder_name = f"{n}best_recog"
new_folder_path = os.path.join(result_path, new_folder_name)

# Ensure the folder exists
os.makedirs(new_folder_path, exist_ok=True)

# Find the file "result.txt" in result_path
source_file = os.path.join(result_path, "result.txt")

if os.path.exists(source_file):
    # Move the file "result.txt" to the folder "{n}best_recog"
    destination_file = os.path.join(new_folder_path, "result.txt")
    shutil.move(source_file, destination_file)
    print(f"Moved {source_file} to {destination_file}")
else:
    print(f"File 'result.txt' not found in {result_path}")
