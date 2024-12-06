from abc import ABC, abstractmethod
from stemmerParser import *

class StemmerWrapper(ABC):
    @abstractmethod
    def stemSentence(self, sent):
        pass

class StemmerRK(StemmerWrapper):
    def __init__(self, stemmerCore: StemmerCore):
        self.stemmerCore = stemmerCore

    def setStemmerCore(self, stemmerCore: StemmerCore):
        self.stemmerCore = stemmerCore

    def stemSentence(self, sent):
        tokens = sent.split()
        stemmed_tokens = []
        for token in tokens:
            stemmed_tokens.append(self.stemmerCore.stem_word(token))
        return " ".join(stemmed_tokens)
    

if __name__ == "__main__":
    wordDict = WordDict()
    priorityRules = {
        "replace": 1,
        "remove": [0,2,3],
        "ambiguous": 4
    }

    stemmerCore = RafiStemmerMod(wordDict, priorityRules)
    stemmerCore2 = RafiStemmer()
    stemmerWrapper = StemmerRK(stemmerCore)
    
    print(stemmerWrapper.stemSentence("পর্যটনলিপি আজ আপনাদের সামনে তুলে ধরবে ঢাকার বুকে অবস্থিত এক ঐতিহাসিক ভবন কার্জন হল।"))

    stemmerWrapper.setStemmerCore(stemmerCore2)

    print(stemmerWrapper.stemSentence("পর্যটনলিপি আজ আপনাদের সামনে তুলে ধরবে ঢাকার বুকে অবস্থিত এক ঐতিহাসিক ভবন কার্জন হল।"))
