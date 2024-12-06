import json
from normalizer import normalize

suffix = json.load(open("suffixCategories.jsonl", "r", encoding="utf-8"))
print(type(suffix))


def convert_to_dict(input_list):
    result_dict = {}
    for sublist in input_list:
        word = sublist[0]
        word = normalize(word)
        value_str = sublist[1]
        keyList = [int(w) for w in value_str.split("/")]
        suffixList = []
        for key in keyList:
            suffixList.extend(suffix[str(key)])
        result_dict[word] = suffixList
    return result_dict


with open("./WeatWords/allTraitWords.txt", "r") as file:
    weatWordList = [w.split(", ") for w in file.read().split("\n")]
    print(weatWordList[-1])
    weatWordDict = convert_to_dict(weatWordList)

with open("traitWordsWithSuffix.jsonl", "w", encoding="utf-8") as f:
    json.dump(weatWordDict, f, ensure_ascii=False)