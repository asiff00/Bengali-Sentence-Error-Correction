import gdown
import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
id = "1GYj3dMejzMsjm838WOnfbFJZ1tLrv5TR"  # Don't worry it's a 'Anyone with the link' access.
dir = "checkpoint"
file = "model.safetensors"


def download_file_from_google_drive(id, dir, file):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f"Created directory at {dir}")
    else:
        print(f"Directory already exists at {dir}")

    file_path = os.path.join(dir, file)
    if not os.path.exists(file_path):
        print(f"File does not exist at {file_path}, downloading now...")
        gdown.download_folder(id=id, output=dir)
        print(f"Downloaded file from Google Drive to {file_path}")
    else:
        print(f"File already exists at {file_path}, skipping download.")


download_file_from_google_drive(id, dir, file)

checkpoint = "checkpoint"
tokenizer = AutoTokenizer.from_pretrained(
    checkpoint, src_lang="bn_IN", tgt_lang="bn_IN", use_fast=True
)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, use_safetensors=True)
model = model.to(device)


while True:
    incorrect_bengali_sentence = str(input("Enter a sentence: "))
    inputs = tokenizer.encode(
        incorrect_bengali_sentence,
        truncation=True,
        return_tensors="pt",
        max_length=len(incorrect_bengali_sentence),
    )

    output_ids = model.generate(
        inputs,
        max_new_tokens=len(incorrect_bengali_sentence),
        early_stopping=True,
    )
    print(
        f"Correct sentence: {tokenizer.decode(output_ids[0], skip_special_tokens=True)}"
    )
