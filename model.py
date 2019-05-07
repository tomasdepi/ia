import pandas as pd
import utilities
import constants
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import UnsupervisedDataSet  
from pybrain.structure.modules import TanhLayer

print('[+] Loading data set')

dataset = SupervisedDataSet(constants.MAX_WORD_LENGHT, 1)

dataframe = pd.read_csv('finalDataSetTrainingWithCharacters.csv')
dataframe.drop(columns=['Word', 'Len'], inplace=True)

for _, row in dataframe.iterrows():
	
	output = tuple([row["Language"]])
	inputs = tuple(row[1:]) # remove language, keep only the 20 characters values
 
	dataset.addSample(inputs, output)

print('[+] Dataset loaded successfully')
print('[+] Starting Tranining.....')

net = buildNetwork(constants.MAX_WORD_LENGHT, 10, 1, bias=True, hiddenclass=TanhLayer) # cant input nodes, hidden nodes nd output nodes
trainer = BackpropTrainer(net, dataset)
trainer.trainUntilConvergence()

print("[+] The Training has finished")

while 1:

	userInput = input("ingrese palabra:")

	if len(userInput) > MAX_WORD_LENGHT:
		print("[-] Error: The word exced {maxCharacters} characters".format(maxCharacters=constants.MAX_WORD_LENGHT))
		continue

	netInput = utilities.wordToList(userInput)
	print(netInput)

	dst = UnsupervisedDataSet(constants.MAX_WORD_LENGHT, )
	dst.addSample(tuple(netInput))
	result=net.activateOnDataset(dst)

	#result = net.activate(netInput)

	print(result)

