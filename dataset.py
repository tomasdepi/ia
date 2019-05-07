import pandas as pd
from pandas.io.json import json_normalize
import json
from unicodedata import *
import constants
import utilities

dictFiles = {constants.SPANISH: "./words/spanish_words.json", 
			 constants.ENGLISH: "./words/english_words.json", 
			 constants.FRENCH: "./words/french_words.json", 
			 constants.GERMAN: "./words/german_words.json"}

datasetTraining = pd.DataFrame()

for language, jsonFile in dictFiles.items():
    with open(jsonFile) as file:
        data = json.load(file)
        
    
    subDataFrame = pd.DataFrame()
    subDataFrame["Word"] = data
    subDataFrame["Language"] = language
    
    datasetTraining = pd.concat([datasetTraining, subDataFrame])
    
    
#add japanese
japaneseDataframe = pd.read_csv("./words/japanese_words.csv", encoding='GBK')
japaneseDataframe = japaneseDataframe[["word1", "word2"]]

secondColumn = japaneseDataframe["word2"]
secondColumn = secondColumn.rename("word1")
japaneseDataframe.drop(columns=["word2"], inplace=True)
japaneseDataframe = pd.concat([japaneseDataframe, secondColumn.to_frame()])
japaneseDataframe.rename(index=str, columns={"word1": "Word"}, inplace=True)
japaneseDataframe["Language"] = constants.JAPANESE
    
 
datasetTraining = pd.concat([datasetTraining, japaneseDataframe])


datasetTraining["Len"] = [len(word) for word in datasetTraining["Word"]]
datasetTraining = datasetTraining.query('Len <= 20')

for x in range(1,21):
    subIndex = x-1
    datasetTraining["c"+str(x)] = [utilities.wordToList(word)[subIndex] for word in datasetTraining["Word"]]

datasetTraining.to_csv("finalDataSetTrainingWithCharacters.csv", index=False)
