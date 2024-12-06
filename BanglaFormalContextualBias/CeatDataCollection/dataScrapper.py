from Stemmer import *
from stemmerParser import RafiStemmer, WordDict
from wordFinder import *
import pybmoore
import os 
import sys 
from tqdm import tqdm

class DataScrapper():
    def __init__(self,
                 filenames: list[str],
                 evaluator: WordFinder,
                 stemmer: StemmerWrapper = None,
                 rootDir: str = "") -> None:
        self.rootDir = rootDir
        self.filenames = filenames
        self.stemmer = stemmer
        self.evaluator = evaluator 
        self.resetLists()

    def resetLists(self):
        self.sentenceList = []
        self.filesIndexList = []
        self.sentSerial = 0
        self.currentFileName = ""

    def lookIntoFile(self, file):
        currentSent = ""
        prevSent = ""
        nextSent = ""
        for line in tqdm(file.readlines(),  desc="Scraping data"):
            line = line.strip()
            if self.stemmer:
                nextSent = self.stemmer.stemSentence(line)
            else:
                nextSent = line
            if currentSent == "":
                currentSent = nextSent
                continue
            
            if self.evaluator.evaluate(currentSent, self.sentSerial):
                sentence = ' '.join([prevSent, currentSent, nextSent])
                self.sentenceList.append(sentence)
                self.sentSerial += 1
                self.filesIndexList.append(self.currentFileName)
            prevSent = currentSent
            currentSent = nextSent

    def scrapeData(self):
        self.resetLists()
        for filename in self.filenames:
            with open(os.path.join(self.rootDir, filename), "r") as file:
                self.currentFileName = filename
                self.lookIntoFile(file)

        return self.evaluator.getWeatWordDict(), self.sentenceList, self.filesIndexList
                

if __name__ == "__main__":
    def get_all_files(directory):
        all_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                all_files.append(file_path)
        return all_files
    
    filesList = []
    for i in range(1, len(sys.argv)):
        files = get_all_files(sys.argv[i])
        filesList.extend(files)

    print(filesList)

    stemmerCore = RafiStemmer()
    stemmerWrapper = StemmerRK(stemmerCore)

    weatWordList = ["প্রকৃতি", "পর্যটনলিপি", "কুয়াকাটা", "বৌদ্ধ", "লালবাগ", "কলেজ"]

    dsc = DataScrapper(filesList, stemmerWrapper, weatWordList)

    print(dsc.scrapeData())