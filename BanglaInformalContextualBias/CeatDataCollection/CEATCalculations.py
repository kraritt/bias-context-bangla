from extractCeatCategories import getCeatWords
from ConfigurationVariables import *
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import norm
import numpy as np
import pickle
import os
import sys
from prettytable import PrettyTable
from tqdm import tqdm
import scipy


def associate(w, A, B):
    return (
        cosine_similarity(w.reshape(1, -1), A).mean()
        - cosine_similarity(w.reshape(1, -1), B).mean()
    )


def difference(X, Y, A, B):
    return np.sum([associate(X[i, :], A, B) for i in range(X.shape[0])]) - np.sum(
        [associate(Y[i, :], A, B) for i in range(Y.shape[0])]
    )


def EffectSize(X, Y, A, B):
    delta_mean = np.mean(
        [associate(X[i, :], A, B) for i in range(X.shape[0])]
    ) - np.mean([associate(Y[i, :], A, B) for i in range(Y.shape[0])])

    XY = np.concatenate((X, Y), axis=0)
    s = [associate(XY[i, :], A, B) for i in range(XY.shape[0])]

    std_dev = np.std(s, ddof=1)
    var = std_dev**2

    return delta_mean / std_dev, var


def CEAT_DataGeneration(
    weatGroup: list,
    embeddingsDict,
    nSample=10000,
    model="bert",
    save=False,
):
    effectSizeArray = np.array([], dtype=np.float32)
    varianceArray = np.array([], dtype=np.float32)

    XSet = weatGroup[0]
    YSet = weatGroup[1]
    ASet = weatGroup[2]
    BSet = weatGroup[3]

    for i in tqdm(range(nSample)):
        X = np.array(
            [
                embeddingsDict[word][np.random.randint(0, len(embeddingsDict[word]))]
                for word in XSet
            ]
        )
        Y = np.array(
            [
                embeddingsDict[word][np.random.randint(0, len(embeddingsDict[word]))]
                for word in YSet
            ]
        )
        A = np.array(
            [
                embeddingsDict[word][np.random.randint(0, len(embeddingsDict[word]))]
                for word in ASet
            ]
        )
        B = np.array(
            [
                embeddingsDict[word][np.random.randint(0, len(embeddingsDict[word]))]
                for word in BSet
            ]
        )

        effectSize, variance = EffectSize(X, Y, A, B)

        effectSizeArray = np.append(effectSizeArray, effectSize)
        varianceArray = np.append(varianceArray, variance)

    if save:
        pickle.dump(effectSizeArray, open("es_" + model + ".pickle", "wb"))
        pickle.dump(variance, open("var_" + model + ".pickle", "wb"))

    return effectSizeArray, varianceArray


def CEAT_MetaAnalysis(
    effectSizeArray, V, nSample=10000
):
    V = V.astype(np.float64)
    W = 1.0 / V
    Q = np.sum(W * (effectSizeArray**2)) - (
        (np.sum(W * effectSizeArray) ** 2) / np.sum(W)
    )

    df = nSample - 1

    if Q >= df:
        C = np.sum(W) - np.sum(W**2) / np.sum(W)
        sigma_square_btn = (Q - df) / C
    else:
        sigma_square_btn = 0

    v = 1 / (V + sigma_square_btn)

    CES = np.sum(v * effectSizeArray) / np.sum(v)

    SE_CES = np.sqrt(1.0 / np.sum(v))

    p_value = 2.0 * norm.sf(np.abs(CES / SE_CES), loc=0, scale=1)

    return CES, p_value


def writeDataValue(model, data, sentenceLengths, nSample):
    table = PrettyTable()
    headers = ["CEAT Type", "Data Value"]
    headers.extend([f"Length: {lenString}" for lenString in sentenceLengths])

    table.field_names = headers

    print(headers)

    for category, value in data.items():
        row = [
            f"{category}\n{value['target']}\n{value['attribute']}",
            "\nCES:\nP-Value:",
        ]
        row.extend(
            [
                f"\n{value[lenString]['CES']}\n{value[lenString]['p']}"
                for lenString in sentenceLengths
            ]
        )
        table.add_row(row)

    with open(f"./results/{model}_ceat_results_{nSample}.txt", "w") as f:
        f.write(table.get_string())
        f.write("\n\n")
        f.close()


if __name__ == "__main__":
    categoryDefinition, ceatData = getCeatWords()

    embeddingsMapper = GetEmbeddingsFileMapping()

    sentenceLengths = GetOperationalSentenceLengths()
    seed = 32
    nSample = 7000
    np.random.seed(seed=seed)
    experimentType = "random"
    if len(sys.argv) >= 2 and sys.argv[1] == "-exp" and sys.argv[2] == "fixed":
        experimentType = "fixed"

    print(f"Experiment Type: {experimentType}")

    for model in embeddingsMapper:
        print(f"Model In Use: {model} Sample Size: {nSample}")
        if experimentType == "fixed":
            np.random.seed(seed=seed)

        embeddingsFileFormat = embeddingsMapper[model]
        data = {}
        
        for category in categoryDefinition:
            data[category["Category Name"]] = {
                "target": category["target(s)"],
                "attribute": category["attribute(s)"],
            }

        availableLengths = []

        for lenString in sentenceLengths:
            embeddingsFileName = embeddingsFileFormat % lenString
            print(f"Embeddings File Name: {embeddingsFileName}")
            embeddingsFilePath = os.path.join("./embeddings", model, embeddingsFileName)
            if not os.path.exists(embeddingsFilePath):
                print("File not found")
                continue
            availableLengths.append(lenString)
            embeddingsDict = pickle.load(open(embeddingsFilePath, "rb"))
            print("Done Loading...")
            for testIndex, ceatGroup in enumerate(ceatData):
                effectSizeArray, varianceArray = CEAT_DataGeneration(
                    ceatGroup,
                    embeddingsDict,
                    nSample=nSample,
                    model=model,
                    save=False,
                )
                print("Saving pickle files...")
                print(
                    f"Description: {categoryDefinition[testIndex]['Category Name']}, Length: {lenString}, Model: {model}"
                )
            
                pes, p_value = CEAT_MetaAnalysis(
                    effectSizeArray, varianceArray, nSample=nSample
                )

                data[categoryDefinition[testIndex]["Category Name"]][lenString] = {
                    "CES": pes,
                    "p": p_value,
                }

        print((("Writing To file...")))
        writeDataValue(
            model=model, data=data, sentenceLengths=availableLengths, nSample=nSample
        )
