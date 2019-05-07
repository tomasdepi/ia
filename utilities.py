def wordToList(word):

	lenWord = len(word)
	listOfCeros = [0] * (20-lenWord)
	l = [ord(character) for character in word]

	return l + listOfCeros