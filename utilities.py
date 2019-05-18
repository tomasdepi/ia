import constants

def wordToList(word):

	lenWord = len(word)
	listOfCeros = [0] * (constants.MAX_WORD_LENGHT-lenWord)
	l = [ord(character) for character in word]

	return l + listOfCeros