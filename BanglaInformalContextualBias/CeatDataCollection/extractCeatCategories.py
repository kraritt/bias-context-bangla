import os
import json
from normalizer import normalize

ceatRootDir = "./CEAT_Categories"


def normalizeList(wordList):
    newWordList = []
    for word in wordList:
        newWordList.append(
            normalize(word, unicode_norm="NFKC", apply_unicode_norm_last=True)
        )
    return newWordList


def getCeatWords():
    categoryDefinition = []
    ceatData = []
    for filename in sorted(os.listdir(ceatRootDir)):
        data = json.load(
            open(os.path.join(ceatRootDir, filename), "r", encoding="utf-8")
        )
        categoryDefinition.append(
            {
                "Category Name": filename.split(".")[0],
                "target(s)": [data["targ1"]["category"], data["targ2"]["category"]],
                "attribute(s)": [data["attr1"]["category"], data["attr2"]["category"]],
            }
        )

        ceatData.append(
            [
                normalizeList(data["targ1"]["examples"]),
                normalizeList(data["targ2"]["examples"]),
                normalizeList(data["attr1"]["examples"]),
                normalizeList(data["attr2"]["examples"]),
            ]
        )

    return categoryDefinition, ceatData


if __name__ == "__main__":
    categoryDefinition, ceatData = getCeatWords()

    for i, category in enumerate(categoryDefinition):
        print(category["Category Name"])
        print("target: ", category["target(s)"])
        print("attribute: ", category["attribute(s)"])
        print("---------------------------------------------------")