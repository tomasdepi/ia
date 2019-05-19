import pandas as pd
import utilities
import constants
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import UnsupervisedDataSet  
from pybrain.structure.modules import TanhLayer
import sys

testName = str(sys.argv[1])

print('[+] Loading data set')

dataset = SupervisedDataSet(constants.MAX_WORD_LENGHT, constants.OUTPUT_NODES)

dataframe = pd.read_csv('{testName}.csv'.format(testName=testName))
dataframeTraining = dataframe.drop(columns=['Word', 'Len'])


dataframeLen = len(dataframe)
lenForTesting = round(dataframeLen * 0.25)

dataframeTesting = dataframeTraining.sample(lenForTesting)
dataframeTraining = dataframeTraining[~dataframeTraining.index.isin(list(dataframeTesting.index))]
dataframeTraining.to_csv("training_{testName}.csv".format(testName=testName), index=False)

for _, row in dataframeTraining.iterrows():
	
	output = tuple([row["Language"]])
	inputs = tuple(row[1:]) # remove language, keep only the 20 characters values

	dataset.addSample(inputs, output)


print('[+] Dataset loaded successfully')
print('[+] Starting Tranining.....')

net = buildNetwork(constants.MAX_WORD_LENGHT, constants.HIDDEN_NODES, constants.OUTPUT_NODES, bias=True) # cant input nodes, hidden nodes nd output nodes
trainer = BackpropTrainer(net, dataset, learningrate = 0.05)
trainer.trainOnDataset(dataset, 10)
trainer.testOnData(verbose=True)
#trainer.train()

print("[+] The Training has finished")

words = []
languageIds = []
languages = []
results = []
errors = []
inputs = [] 
for rowId, row in dataframeTesting.iterrows():

	output = tuple([row["Language"]])
	inputNet = tuple(row[1:]) # remove language, keep only the 20 characters values	
	result = net.activate(inputNet)[0].round(5)
	languageId = dataframe.at[rowId, "Language"]
	error = (result - languageId).round(5)
	word = dataframe.at[rowId, "Word"]

	words.append(word)
	languageIds.append(languageId)
	languages.append(constants.dicLanguageIdToString[languageId])
	results.append(result)
	errors.append(error)
	inputs.append(inputNet)

dfSaveOutput = pd.DataFrame(data={'Word':words, "Input":inputs, 'Expected Language Id': languageIds, 'Expected Language': languages, 'Result': results, 'Error': errors})
dfSaveOutput.to_csv('result_{testName}.csv'.format(testName=testName), index=False)

