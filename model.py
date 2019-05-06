import pandas as pd
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import UnsupervisedDataSet  
from pybrain.structure.modules import TanhLayer

MAX_WORD_LENGHT = 20

def wordToList(word):

	lenWord = len(word)
	listOfCeros = [0] * (20-lenWord)
	l = [ord(character) for character in word]

	return l + listOfCeros

print('[+] Loading data set')

dataset = SupervisedDataSet(MAX_WORD_LENGHT, 1)

dataframe = pd.read_csv('finalDataSetTrainingWithCharacters.csv')
dataframe.drop(columns=['Word', 'Len'], inplace=True)

for _, row in dataframe.iterrows():
	
	output = tuple([row["Language"]])
	inputs = tuple(row[1:]) # remove language, keep only the 20 characters values
 
	dataset.addSample(inputs, output)

print('[+] Dataset loaded successfully')
print('[+] Starting Tranining.....')

net = buildNetwork(MAX_WORD_LENGHT, 10, 1, bias=True, hiddenclass=TanhLayer) # cant input nodes, hidden nodes nd output nodes
trainer = BackpropTrainer(net, dataset)
trainer.trainUntilConvergence()

print("[+] The Training has finished")

while 1:

	userInput = input("ingrese palabra:")

	if len(userInput) > MAX_WORD_LENGHT:
		print("[-] Error: The word exced {maxCharacters} characters".format(maxCharacters=MAX_WORD_LENGHT))
		continue

	netInput = wordToList(userInput)
	print(netInput)

	dst = UnsupervisedDataSet(MAX_WORD_LENGHT, )
	dst.addSample(tuple(netInput))
	result=net.activateOnDataset(dst)

	#result = net.activate(netInput)

	print(result)

