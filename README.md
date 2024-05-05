# Project Overview: Bengali Text Correction

The goal of this project was to develop a software model that could fix grammatical and syntax errors in Bengali text. The approach was similar to how a language translator works, where the incorrect sentence is transformed into a correct one. We fine tune a pertained model, namely [mBart50](https://huggingface.co/facebook/mbart-large-50) with a [dataset](https://github.com/hishab-nlp/BNSECData) of 1.3 M samples for 6500 steps and achieve a score of `{BLEU: 0.443, CER:0.159, WER:0.406, Meteor: 0.655}`when tested on unseen data. Clone/download this repo, run the `correction.py` script and type the sentence after the prompt and you are all set.

## Initial Testing:

At the beginning, I experimented with several T5 models (mt5(small/base), Flan T5, and Bengali T5), but the results were not very good. I was limited by computational resources, which allowed me to train these models with only 10% of the data for 1 to 2 epochs. With this limited testing the result wasn't promising enough to invest all the available resources to the limit therefore I explored other models and found a winner model that is well suited for this task.

I also tested casual large models like Mistral 7B and Gemma 2B, and even with optimizations like QLoRa, they were too large and costly to run.

During the initial testing, I tried training the same models different token lengths, a maximum token length of 20 provided much better results than 64. The current model has a maximum token length of 32.

Beyond Seq2Seq models and approach a few other ideas also crossed my mind. Other methods considered included using NER (Named Entity Recognition) to tag words as correct or incorrect, and masked models that focused on correcting one wrong word at a time. Both methods required knowing the errors in advance or making multiple calls to get a final verdict, which was not practical to say. There are other solutions too that doesn't use ML at all. Approaches like running each words against a reference list and replace them when there's no hit. Attempts to replace each word based on a reference list worked somewhat like a spell checker which wasn't the goal.

Ultimately, mBART 50 was chosen as the best model because of its flexibility, resource efficiency, and reproducibility.

## Base Model Overrides

[mBART Large 50](https://huggingface.co/facebook/mbart-large-50) is a 600M parameter multilingual Sequence-to-Sequence model. It was introduced to show that multilingual translation models can be created through multilingual fine-tuning. Instead of fine-tuning on one direction, a pre-trained model is fine-tuned on many directions simultaneously. mBART-50 is created using the original mBART model and extended to add extra 25 languages to support multilingual machine translation models of 50 languages. More about the base model can be found in [Official Documentation](https://huggingface.co/docs/transformers/model_doc/mbart)

## Data Processing and Enhancements

[Dataset](https://github.com/hishab-nlp/BNSECData) contains over 1.3M incorrect, correct sentences pair. When reviewing the data, I found some outliers, such as multiple occurrences of the digit '1'. Instead of removing these, I compressed them into a single number to help the model understand numbers in context. To simulate common user errors, I generated additional incorrect sentences with errors custom errors. One of them would be homophonic character swaps and another diacritic changes. This is one the common mistake in bengali writing. Here, `পরি`/`পড়ি` and `বিশ`, `বিষ` pair has completely different meaning depending on which spelling we are using. So, it is important that out dataset acknowledges that. I created a [Simulate Error](simulate_error.py) script for this project where you can add easily add introduce errors to your bengali dataset.

After necessary data engineering, I split the dataset into train and test sets. However, looking back at the choice of distribution (test_size=0.005) , which was purely to accommodate computational scarcity, I realize I have made a mistake there.

## Fine tuning:

The model was fine tuned with the help of Transformers framework. It was trained on a dataset of 1.3M correct/incorrect sentence pairs which was distributed to 1,349,518 training samples and 6,782 validation samples over 6500 global steps. Configured for a max token length of 32, the training was conducted on a Google Colab A100 GPU for one epoch with a batch size of 128. The model used the AdamW optimizer with a 1e-5 learning rate, tracking metrics like BLEU, ROUGE, WER, and CER. An early stopping mechanism safeguarded against overfitting. The best model, determined by BLEU score, was saved for future use. I included the entire step by step fine tune and training process in here [Fine Tune](finetune.ipynb)

## Model Performance Metrics

| Metric | Training | Post-Training Testing |
| ------ | -------- | --------------------- |
| BLEU   | 0.805    | 0.443                 |
| CER    | 0.053    | 0.159                 |
| WER    | 0.101    | 0.406                 |
| Meteor | 0.904    | 0.655                 |

Testing on unknown data dataset shows a BLEU score of 0.443 which is significantly lower than the BLEU score during the training. We overfit the model.

## Final Implementation and Usage

Here is a simple way to use the fine-tuned model to correct Bengali sentences:
If your are trying to use it on a script, this is how can do It:

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

checkpoint = "model/checkpoint"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, src_lang="bn_IN", tgt_lang="bn_IN", use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint,  use_safetensors =True)

incorrect_bengali_sentence = "আপনি কমন আছেন?"
inputs = tokenizer.encode(incorrect_bengali_sentence, truncation = True, return_tensors='pt', max_length=len(incorrect_bengali_sentence))
outputs = model.generate(inputs, max_new_tokens=len(incorrect_bengali_sentence), num_beams=5, early_stopping=True)
correct_bengali_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
# আপনি কেমন আছেন?
```

If you want to test this model from the terminal, run the `python correction.py` script and type the sentence after the prompt and you are all set. you'll need the `transformers` library to run this script. Install the `transformers` model using `pip install -q transformers[torch] -U`.

#### Important note: You need to make sure if have used `use_safetensors =True` parameter during loading the model.

# General issues faced during the entire journey:

- Issue: The system is not printing any evaluation function.
  Solution: The GPU that I am training on doesn't support FP16/BF16 precision. Commenting out `fp16 =True` in the Seq2SeqTrainingArguments solved the issue.

- Issue: Training on TPU crashes on both Colab and Kaggle.
  Solution: See https://github.com/pytorch/xla/issues/6990#issuecomment-2083770632 for more information.

## What's next?

The model is clearly overfitting, and we can reduce that. My best guess is that we have a comparatively smaller validation set, which needed to be smaller to fit the model on a GPU, thus exacerbating the huge discrepancy between the two tests. We can train it on a more balanced distribution of datasets for further improvement. Another thing we can do is fine-tune the already fine-tuned model using a new dataset. I already have a script, [Scrapper](https://github.com/himisir/Scrape-Any-Sites), that I can use with the [Data Pipeline](simulate_error.py) that I just created for more diverse training data.

I'm also planning to run a 4-bit quantization on the same model to see how it performs against the base model. It should be a fun experiment.

## Resources and References:

[Dataset Source](https://github.com/hishab-nlp/BNSECData)
[Model Documentation and Troubleshooting](https://huggingface.co/docs/transformers/model_doc/mbart)
