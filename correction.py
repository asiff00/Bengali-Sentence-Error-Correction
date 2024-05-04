import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = "model/checkpoint"
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
