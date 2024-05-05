import torch
from download_model import download_file_from_google_drive
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

id = "1GYj3dMejzMsjm838WOnfbFJZ1tLrv5TR"  # Don't worry it's a 'Anyone with the link' access.
dir = "checkpoint"
file = "model.safetensors"


def model_call(checkpoint):
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, src_lang="bn_IN", tgt_lang="bn_IN", use_fast=True
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, use_safetensors=True)
    model = model.to(device)
    return tokenizer, model


def chat_loop(tokenizer, model):  # Chat loop! enter 'quit' to quit
    while True:
        given_sentence = str(input("Enter a sentence: "))
        if given_sentence.lower() == "quit":
            break
        inputs = tokenizer.encode(
            given_sentence,
            truncation=True,
            return_tensors="pt",
            max_length=len(given_sentence),
        )

        output_ids = model.generate(
            inputs,
            max_new_tokens=len(given_sentence),
            early_stopping=True,
        )
        print(
            f"Correct sentence: {tokenizer.decode(output_ids[0], skip_special_tokens=True)}"
        )


try:
    c = 1 / 0  # Intentional, currently there's an error, so it may not work.
    checkpoint = "asif00/mbart_bn_error_correction"  # Currently there's an error, so it may not work.
    tokenizer, model = model_call(checkpoint)
    chat_loop(tokenizer, model)

except (
    Exception
) as e:  # Backup files from my google drive, 2GB download, may take a few minutes
    checkpoint = "checkpoint"
    download_file_from_google_drive(id, dir, file)
    tokenizer, model = model_call(checkpoint)
    chat_loop(tokenizer, model)
