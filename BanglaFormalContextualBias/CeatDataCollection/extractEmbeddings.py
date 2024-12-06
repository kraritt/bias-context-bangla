import sys
from wordFinder import *
import json
from extractSentences import normalizeWeatDict
from Stemmer import *
import numpy as np
import random
from tqdm import tqdm
import pickle
from models import ModelWrapper, MLMEmbeddingExtractor, BanglaBertDiscriminator


def getSentencesSample(sentenceList, maxItems=1000):
    numItems = min(maxItems, len(sentenceList))
    items = (
        sentenceList if numItems < maxItems else random.sample(sentenceList, maxItems)
    )
    return items


class SentenceProcessor:
    def __init__(self, evaluator: WordFinder, stemmer: StemmerWrapper = None) -> None:
        self.evaluator = evaluator
        self.stemmer = stemmer

    def setLength(self, length: int) -> None:
        self.length = length

    def shortenSentence(self, sentence: str, word: str) -> str:
        indices = self.evaluator.getIndex(sentence, word)
        if len(indices) == 0:
            raise Exception("No match found")

        index = indices[0]
        if self.length == -1:
            return sentence, index
        words = sentence.split()
        modifiedIndex = index
        if len(words) >= self.length:
            if index < self.length // 2:
                wordsUsed = words[: self.length]
            elif (len(words) - index) < self.length // 2 + 1:
                wordsUsed = words[-self.length :]
                modifiedIndex = self.length - (len(words) - index)
            else:
                wordsUsed = words[index - self.length // 2 : index + self.length // 2]
                modifiedIndex = self.length // 2
            newSentence = " ".join(wordsUsed)
        else:
            newSentence = sentence

        if self.stemmer:
            newSentence = self.stemmer.stemSentence(newSentence)
        return newSentence, modifiedIndex

    def getSpan(self, sentence: str, word: str, index: int) -> list[int]:
        return self.evaluator.getSpanByIndex(sentence, word, index)


class EmbeddingExtractor:
    def __init__(
        self, sentenceProcessor: SentenceProcessor, model: ModelWrapper, loggerFile=None
    ) -> None:
        self.sentenceProcessor = sentenceProcessor
        self.model = model
        self.loggerFile = loggerFile
        self.randomSelect = False
        self.maxSentenceSample = 100000

    def setLoggerFile(self, loggerFile) -> None:
        self.loggerFile = loggerFile

    def setRandomSelect(self, randomSelect: bool) -> None:
        self.randomSelect = randomSelect

    def setMaxSentenceSample(self, maxSentenceSample: int) -> None:
        self.maxSentenceSample = maxSentenceSample

    def getSampleSentences(self, sentenceList: list[str]) -> list[str]:
        sentences = []
        if self.randomSelect:
            if len(sentenceList) > self.maxSentenceSample:
                sentences = random.sample(sentenceList, self.maxSentenceSample)
            else:
                sentences = sentenceList
        else:
            sentenceRange = min(len(sentenceList), self.maxSentenceSample)
            sentences = sentenceList[:sentenceRange]

        return sentences

    def extract(
        self, weatWordSentenceDict: dict[str, list[str]]
    ) -> dict[str, list[np.array]]:
        weatWordEmbeddings = {}

        for word in weatWordSentenceDict:
            weatWordEmbeddings[word] = []

        for word in weatWordSentenceDict:
            print(f"Processing For: {word}")

            sentences = self.getSampleSentences(weatWordSentenceDict[word])
            for i, sentence in tqdm(
                enumerate(sentences),
                desc="Processing Sentences",
            ):
                try:
                    sentence, index = self.sentenceProcessor.shortenSentence(
                        sentence, word
                    )
                except:
                    if self.loggerFile:
                        self.loggerFile.write(
                            f"Cannot find {word} at {index}\nSentence: {sentence}\n"
                        )
                    continue
                span = self.sentenceProcessor.getSpan(sentence, word, index)
                try:
                    weatWordEmbeddings[word].append(
                        self.model.getWordVector(word, sentence, index, span)
                    )
                except:
                    if self.loggerFile:
                        self.loggerFile.write(
                            f"Error for {word} at {index}\nSentence: {sentence}\n"
                        )

        return weatWordEmbeddings


if __name__ == "__main__":
    weatWordDict = json.load(open("weatWordsWithSuffix.jsonl", "r", encoding="utf-8"))
    weatWordDict = normalizeWeatDict(weatWordDict)
    evaluator = WordEvaluatorRegexSuffixFixed(weatWordDict)
    processor = SentenceProcessor(evaluator)

    if sys.argv[1] == "-l":
        sentenceLength = int(sys.argv[2])
    else:
        print("sentenceLength not defined")
        exit()
    processor.setLength(sentenceLength)

    if sentenceLength == -1:
        nameExtension = "all"
    elif sentenceLength > 0:
        nameExtension = str(sentenceLength)

    models = [
        "BanglaBert_Generator",
        "BanglaBert_Discriminator",
        "Muril_Large",
        "XLM_Roberta_Large",
    ]

    if sys.argv[3] == "-m":
        modelName = models[int(sys.argv[4])]
        print(f"Model: {modelName}")
    else:
        print("Model not defined")
        exit()

    if modelName == "BanglaBert_Generator":
        model = MLMEmbeddingExtractor(
            model_name="csebuetnlp/banglabert_large_generator",
            tokenizer_name="csebuetnlp/banglabert_large_generator",
        )
    elif modelName == "BanglaBert_Discriminator":
        model = BanglaBertDiscriminator(
            model_name="csebuetnlp/banglabert_large",
            tokenizer_name="csebuetnlp/banglabert_large",
        )
    elif modelName == "Muril_Large":
        model = MLMEmbeddingExtractor(
            model_name="google/muril-large-cased",
            tokenizer_name="google/muril-large-cased",
        )
    elif modelName == "XLM_Roberta_Large":
        model = MLMEmbeddingExtractor(
            model_name="xlm-roberta-large",
            tokenizer_name="xlm-roberta-large",
        )

    seed = 47
    random.seed(seed)

    extractor = EmbeddingExtractor(processor, model)

    weatWordSentenceDict = pickle.load(open("./results/result_final_v2.pkl", "rb"))

    if sentenceLength > 0:
        loggerFile = open(f"./embeddings/{modelName}_log_{nameExtension}.txt", "w")
        extractor.setLoggerFile(loggerFile)
        embedding = extractor.extract(weatWordSentenceDict)
        pickle.dump(
            embedding,
            open(f"./embeddings/embeddings_{modelName}_len_{nameExtension}.pkl", "wb"),
        )
        loggerFile.close()
    elif sentenceLength == -2:
        print("Variable Length Run...")
        sentenceLengths = [9, 15, 25, 40, 60, 75, 100, 125, 150, 200]
        for length in sentenceLengths:
            print(f"Length: {length}")
            random.seed(seed)
            nameExtension = str(length)
            processor.setLength(length)
            loggerFile = open(f"./embeddings/{modelName}_log_{nameExtension}.txt", "w")

            extractor.setLoggerFile(loggerFile)
            extractor.setMaxSentenceSample(1200)
            extractor.setRandomSelect(True)

            embedding = extractor.extract(weatWordSentenceDict)
            pickle.dump(
                embedding,
                open(
                    f"./embeddings/embeddings_{modelName}_len_{nameExtension}.pkl", "wb"
                ),
            )

            loggerFile.close()