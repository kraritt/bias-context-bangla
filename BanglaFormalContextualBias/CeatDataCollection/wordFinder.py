from abc import ABC, abstractmethod
import pybmoore
import re
from normalizer import normalize


class WordFinder(ABC):
    def __init__(self, weatWordList: list[str]):
        self.weatWordList = weatWordList
        self.weatWordDict = {}
        for word in weatWordList:
            self.weatWordDict[word] = []

    def getWeatWordDict(self):
        return self.weatWordDict

    @abstractmethod
    def evaluate(self, sent: str, serial: int) -> bool:
        pass

    @abstractmethod
    def getIndex(self, sentence, keyWord) -> list[int]:
        pass

    @abstractmethod
    def getSpan(self, sent: str, keyWord: str) -> list[int]:
        pass

    @abstractmethod
    def getSpanByIndex(self, sent: str, keyWord: str, index: int) -> list[int]:
        pass


class WordEvaluatorBMoore(WordFinder):
    def __init__(self, weatWordList: list[str]):
        super().__init__(weatWordList)

    def evaluate(self, sent: str, serial: int) -> bool:
        isMatchFound = False
        for word in self.weatWordList:
            matches = pybmoore.search(word, sent)
            if len(matches) >= 1:
                isMatchFound = True
                self.weatWordDict[word].append(serial)
        return isMatchFound


class WordEvaluatorRegex(WordFinder):
    def __init__(self, weatWordList: list[str]):
        super().__init__(weatWordList)
        self.weatWordPatterns = {}

        for word in weatWordList:
            self.weatWordPatterns[word] = r"\b" + word + r"\w*"

    def evaluate(self, sent: str, serial: int) -> bool:
        isMatchFound = False
        for word in self.weatWordList:
            pattern = self.weatWordPatterns[word]
            matches = re.match(pattern, sent)
            if matches:
                isMatchFound = True
                self.weatWordDict[word].append(serial)
        return isMatchFound


class WordEvaluatorRegexSuffix(WordFinder):
    def __init__(self, weatWordList: list[str], suffixList: list[str]):
        super().__init__(weatWordList)
        self.weatWordPatterns = {}

        suffixString = "|".join(suffixList)

        for word in weatWordList:
            self.weatWordPatterns[word] = r"\b" + word + f"(?:{suffixString})?" + r"\W"

    def evaluate(self, sent: str, serial: int) -> bool:
        isMatchFound = False
        for word in self.weatWordList:
            pattern = self.weatWordPatterns[word]
            matches = re.match(pattern, sent)
            if matches:
                isMatchFound = True
                self.weatWordDict[word].append(serial)
        return isMatchFound


class WordEvaluatorRegexSuffixFixed(WordFinder):
    def __init__(self, weatWordDict: dict[str, list[str]]):
        super().__init__(weatWordDict.keys())
        self.weatWordPatterns = {}
        end_characters = r" ,।:@;\'\"!#\$%\^&\~\-\+\?><\(\)"
        for word in weatWordDict:
            suffixString = "|".join(weatWordDict[word])
            self.weatWordPatterns[word] = (
                r"\b"
                + re.escape(word)
                + f"(?:{suffixString})?"
                + r"["
                + re.escape(end_characters)
                + r"]"
            )

    def evaluate(self, sent: str, serial: int) -> bool:
        isMatchFound = False
        for word in self.weatWordList:
            pattern = self.weatWordPatterns[word]
            matches = re.match(pattern, sent)
            if matches:
                isMatchFound = True
                self.weatWordDict[word].append(serial)
        return isMatchFound

    def getIndex(self, sentence, keyWord) -> list[int]:
        pattern = self.weatWordPatterns[keyWord]
        indices = []
        for i, word in enumerate(sentence.split()):
            if re.match(pattern, word) or re.match(pattern, word + " "):
                indices.append(i)
        return indices

    def getSpan(self, sent: str, keyWord: str):
        pattern = self.weatWordPatterns[keyWord]
        normalizedSent = normalize(sent)
        matches = re.search(pattern=keyWord, string=normalizedSent)
        if matches:
            return matches.span()
        else:
            return re.search(pattern=pattern, string=normalizedSent).span()

    def getSpanByIndex(self, sent: str, keyWord: str, index: int):
        currentPos = 0
        normalizedSent = normalize(sent)
        for i, word in enumerate(normalizedSent.split()):
            if i == index:
                return currentPos, currentPos + len(keyWord)
            else:
                currentPos += len(word) + 1