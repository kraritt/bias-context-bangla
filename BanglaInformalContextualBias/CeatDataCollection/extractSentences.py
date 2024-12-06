from dataScrapper import *
import pandas as pd
from wordFinder import *
import json
from normalizer import normalize


def get_all_files(directory):
    all_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)
    return all_files


def getWeatWords(filepath):
    with open(filepath, "r") as file:
        return [w.rstrip(", ") for w in file.read().split("\n")]


def getSuffixList(filepath):
    with open(filepath, "r") as file:
        return [w.rstrip(", ") for w in file.read().split("\n")]


def normalizeWeatDict(weatWordDict):
    newWeatWordDict = {}
    for word in weatWordDict:
        normalizedWord = normalize(
            word, unicode_norm="NFKC", apply_unicode_norm_last=True
        )
        newList = []
        for suffix in weatWordDict[word]:
            normalizedSuffix = normalize(
                suffix, unicode_norm="NFKC", apply_unicode_norm_last=True
            )
            newList.append(normalizedSuffix)
        newWeatWordDict[normalizedWord] = newList
    return newWeatWordDict


if __name__ == "__main__":
    filesList = []
    dir = False
    if sys.argv[1] == "-f":
        filesList = sys.argv[2:]
    elif sys.argv[1] == "-dir":
        dir = True
        for i in range(2, len(sys.argv)):
            files = get_all_files(sys.argv[i])
            filesList.extend(files)
    else:
        print("Invalid argument")
        exit(1)

    weatWordDict = json.load(open("traitWordsWithSuffix.jsonl", "r", encoding="utf-8"))
    weatWordDict = normalizeWeatDict(weatWordDict)
    weatWordList = list(weatWordDict.keys())
    evaluator = WordEvaluatorRegexSuffixFixed(weatWordDict)
    scrapper = DataScrapper(filesList, evaluator=evaluator)

    weatWordDict, sentenceList, filesIndexList = scrapper.scrapeData()

    saveData = {"WEAT word": [], "Sentences": []}
    for word in weatWordList:
        values = weatWordDict[word]
        value_str = "-".join(str(i) for i in values)
        saveData["WEAT word"].append(word)
        saveData["Sentences"].append(value_str)
    saveData["Index"] = range(len(weatWordList))
    weatWordDF = pd.DataFrame(saveData)

    if dir:
        sentencesDF = pd.DataFrame(
            {
                "Index": range(len(sentenceList)),
                "Sentence": sentenceList,
                "SourceFile": filesIndexList,
            }
        )
    else:
        sentencesDF = pd.DataFrame(
            {"Index": range(len(sentenceList)), "Sentence": sentenceList}
        )

    if dir:
        weatWordDF.to_csv("traitWordsSentences.csv", index=False)
        sentencesDF.to_csv("sentences.csv", index=False)
    else:
        import os

        folderName = "_".join(
            [fileName.split("/")[-1].split(".")[0] for fileName in filesList]
        )
        folderName = folderName + "_trait"
        print(folderName)
        os.makedirs("./" + folderName, exist_ok=True)
        weatWordDF.to_csv(f"./{folderName}/traitWordsSentences.csv", index=False)
        sentencesDF.to_csv(f"./{folderName}/sentences.csv", index=False)