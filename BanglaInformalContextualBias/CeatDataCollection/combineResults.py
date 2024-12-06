import os
import pandas as pd
import pickle
from tqdm import tqdm
import time
import re


def findResultFolders(directoryPath, searchFilename):
    resultFolders = []
    for root, _, files in os.walk(directoryPath):
        if searchFilename in files:
            resultFolders.append(root)
    return resultFolders


def create_word_sentence_dict(sentences_file, words_file):
    sentences_df = pd.read_csv(sentences_file)

    words_df = pd.read_csv(words_file)

    word_sentence_dict = {}

    for _, row in words_df.iterrows():
        word = row["WEAT word"]
        word_sentence_dict[word] = list()

    for index, row in words_df.iterrows():
        word = row["WEAT word"]
        if pd.isna(row["Sentences"]):
            continue
    
        sentence_ids = row["Sentences"].split("-")

        sentences_list = [
            sentences_df.loc[int(sid)]["Sentence"] for sid in sentence_ids
        ]

        word_sentence_dict[word].extend(sentences_list)

    return word_sentence_dict


def merge_dictionaries(main_dict, temp_dict):
    for key, value in temp_dict.items():
        if key in main_dict:
            main_dict[key].extend(value)
        else:
            main_dict[key] = value

    return main_dict


def generate_results(weat_word_dict, dump_reuslts=True):
    os.makedirs("results", exist_ok=True)

    sorted_keys = sorted(
        weat_word_dict.keys(), key=lambda k: len(weat_word_dict[k]), reverse=True
    )

    with open("results/results_trait.txt", "w", encoding="utf-8") as file:
        for key in sorted_keys:
            file.write(f"{key}: {len(weat_word_dict[key])}\n")

    if dump_reuslts:
        pickle.dump(weat_word_dict, open("results/results_trait.pkl", "wb"))


def merge_same_words(
    word_sentence_dict: dict[str, list[str]], same_words_dict: dict[str, list[str]]
) -> dict[str, list[str]]:
    
    for masterWord, value in same_words_dict.items():
        new_list = []
        for word in value:
            if word in word_sentence_dict:
                old_list = word_sentence_dict[word]
                for sent in old_list:
                    new_sent = re.sub(word, masterWord, sent)
                    new_list.append(new_sent)
                word_sentence_dict.pop(word)
        word_sentence_dict[masterWord].extend(new_list)
    return word_sentence_dict


word_sentence_dict = {}

directory_path = "./"
file_name = "sentences.csv"
folders_with_file = findResultFolders(directory_path, file_name)

print("The follwing folders are being combined:")
for folder in folders_with_file:
    print(folder)

for folder in tqdm(folders_with_file, desc="Processing"):
    sentences_file = os.path.join(folder, "sentences.csv")
    weatWords_file = os.path.join(folder, "weatWordsSentences.csv")

    temp_word_sentence_dict = create_word_sentence_dict(sentences_file, weatWords_file)
    word_sentence_dict = merge_dictionaries(word_sentence_dict, temp_word_sentence_dict)

generate_results(word_sentence_dict)