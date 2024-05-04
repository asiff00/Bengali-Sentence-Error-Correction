import json
import random
import re
import string
import pandas as pd

output_dir = "./"


def replace_with_homophones(word):
    adjacent_keys = {
        "অ": "আও",
        "আ": "অও",
        "ই": "ঈউই",
        "ঈ": "ইঈ",
        "উ": "ঊউই",
        "ঊ": "উঊ",
        "ঋ": "ঋ",
        "এ": "ঐএই",
        "ঐ": "এঐই",
        "ও": "ঔঅও",
        "ঔ": "ওঔ",
        "ক": "খগ",
        "খ": "কগ",
        "গ": "ঘগ্",
        "ঘ": "গগ্",
        "ঙ": "ঙং",
        "চ": "ছজ",
        "ছ": "চজ",
        "জ": "ঝয",
        "ঝ": "জয",
        "ঞ": "ঞম",
        "ট": "ঠড",
        "ঠ": "টডথ",
        "ড": "ঢদধ",
        "ঢ": "ডদধ",
        "ণ": "ণনম",
        "ত": "থদত",
        "থ": "তদদ্",
        "দ": "ধড",
        "ধ": "দড",
        "ন": "ণম",
        "প": "ফব",
        "ফ": "প",
        "ব": "ভব্",
        "ভ": "ব",
        "ম": "মন",
        "য": "জঝ",
        "র": "লর্যড়ঢ়য়",
        "ল": "রল",
        "শ": "সষ",
        "ষ": "শস",
        "স": "শষ",
        "হ": "হ্",
        "ড়": "ঢ়য়র",
        "ঢ়": "ড়য়র",
        "য়": "ড়ঢ়্",
        "ৎ": "ৎ্তট",
        "ং": "ঙ্",
        "ঃ": "ঃ্",
        "ঁ": "ঁ্",
    }
    diacritic_mapping = {
        "া": "িীুূৃেৈোৌ",
        "ি": "ীাুূ",
        "ী": "িাুূ",
        "ু": "ূিীা",
        "ূ": "ুিীা",
        "ৃ": "েৈা",
        "ে": "ৈৃো",
        "ৈ": "েৃো",
        "ো": "ৌেৈা",
        "ৌ": "োেৈা",
    }

    idx = random.randint(0, len(word) - 1)
    char = word[idx]

    if char in adjacent_keys:
        word = word[:idx] + random.choice(adjacent_keys[char]) + word[idx + 1 :]
        return word
    elif char in diacritic_mapping:
        new_diacritic = random.choice(diacritic_mapping[char])
        word = word[:idx] + new_diacritic + word[idx + 1 :]
    return word


def swap_adjacent_chars(word):
    if len(word) < 2:
        return word
    idx = random.randint(0, len(word) - 2)
    return word[:idx] + word[idx + 1] + word[idx] + word[idx + 2 :]


def remove_char(word):
    if len(word) < 2:
        return word
    idx = random.randint(0, len(word) - 1)
    return word[:idx] + word[idx + 1 :]


def insert_char(word):
    idx = random.randint(0, len(word))
    char = random.choice(string.ascii_lowercase)
    return word[:idx] + char + word[idx:]


def combine_words(words):
    idx = random.randint(0, len(words) - 2)
    words[idx] = words[idx] + words[idx + 1]
    del words[idx + 1]
    return words


def transpose_char(word):
    if len(word) < 2:
        return word
    idx = random.randint(0, len(word) - 2)
    word = word[:idx] + word[idx + 1] + word[idx] + word[idx + 2 :]
    return word


def repeat_char(word):
    if len(word) < 1:
        return word
    idx = random.randint(0, len(word) - 1)
    word = word[:idx] + word[idx] + word[idx] + word[idx + 1 :]
    return word


def remove_diacritic(word):
    diacritics = "ািীুূৃেৈোৌ"
    new_word = ""
    for char in word:
        if char in diacritics and random.random() < 0.5:
            continue
        new_word += char
    return new_word if new_word else word


def replace_wrong_diacritic(word):
    wrong_diacritic = {
        "া": "ে",
        "ি": "ী",
        "ী": "ি",
        "ু": "ূ",
        "ূ": "ু",
    }
    new_word = ""
    for char in word:
        if char in wrong_diacritic and random.random() < 0.5:
            new_word += wrong_diacritic[char]
        else:
            new_word += char
    return new_word


# This is a helper function
def modify_word_based_on_error_type(word, error_type):
    if error_type == "swap":
        return swap_adjacent_chars(word)
    elif error_type == "remove":
        return remove_char(word)
    elif error_type == "insert":
        return insert_char(word)
    elif error_type == "adjacent":
        return replace_with_homophones(word)
    elif error_type == "combine":
        return word
    elif error_type == "transpose":
        return transpose_char(word)
    elif error_type == "repeat":
        return repeat_char(word)
    elif error_type == "remove_diacritic":
        return remove_diacritic(word)
    elif error_type == "replace_wrong_diacritic":
        return replace_wrong_diacritic(word)


# This is where the real corruption happens.
def introduce_errors(query, error_rate):
    words = query.split()
    if len(words) == 0:
        return query
    num_errors = random.randint(0, 1)  # int((len(words) - 1) * (error_rate - 0.5)))
    for _ in range(num_errors):
        if random.random() < error_rate:
            idx = random.randint(0, len(words) - 1)
            error_types = [
                "swap",
                "remove",
                "insert",
                "adjacent",
                "combine",
                "transpose",
                "repeat",
                "remove_diacritic",
                "replace_wrong_diacritic",
            ]
            error_type = random.choice(error_types)
            if error_type == "combine" and len(words) > 1:
                words = combine_words(words)
            else:
                words[idx] = modify_word_based_on_error_type(words[idx], error_type)
    return " ".join(words)


def create_data_pairs(input_file, target_file, error_rate):
    pairs = []
    for i, t in zip(input_file, target_file):
        if not i.strip() and t.strip():
            continue
        erroneous_query = introduce_errors(i, error_rate)
        pairs.append((erroneous_query, t))
    return pairs


def extract(data, dataset_type, column):
    target_data = data[column]  # target,input
    target_data = target_data.dropna()

    def replace_multiple_digits(text):
        return re.sub(r"(\d)\1+", r"\1", text)

    cleaned_target_data = target_data.apply(replace_multiple_digits)
    cleaned_target_list = cleaned_target_data.tolist()
    json_data = json.dumps(cleaned_target_list, ensure_ascii=False)
    with open(f"{dataset_type}_{column}.json", "w", encoding="utf-8") as f:
        f.write(json_data)


def get_json(path):
    with open(path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
        return json_data


def create_dataset(path, dataset_type, error_rate):
    """
    Create a dataset with common Bengali errors in sentences for NLP task from a specified CSV file.

    This function reads data from a CSV file, extracts specified columns to generate input and target JSON files,
    then combines these files into a single dataset while potentially introducing errors at a specified rate
    for simulation or testing purposes.

    Parameters:
    - path (str): The file path to the CSV data file.
    - dataset_type (str): The type of the dataset which can be 'eval', 'train', or 'test'.
                           This influences the naming of output files and potential processing differences.
    - error_rate (float): A float value between 0 and 1 indicating the proportion of the data that should
                          include simulated errors. This is useful for testing the robustness of models.

    Returns:
    - dataset (dict): A dictionary containing the processed dataset ready to be used for training,
                      testing, or evaluation depending on the `dataset_type`.

    Example usage:
    dataset = create_dataset("path/to/data.csv", "train", 0.1)
    """
    data = pd.read_csv(path)
    print(f"Data shape: {data.shape}")

    extract(data, dataset_type, "Input")
    extract(data, dataset_type, "Target")

    input_file = get_json(f"{output_dir}_{dataset_type}_Input.json")
    target_file = get_json(f"{output_dir}_{dataset_type}_Target.json")

    dataset = create_data_pairs(input_file, target_file, error_rate)

    with open(f"{dataset_type}_data.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False)

    return dataset
