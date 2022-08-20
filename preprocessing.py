import sys
import string

class Preprocessing():
    def __init__(self, trainPath, validPath, testPath, context_size): 
        print("Reading words...")
        self.trainWords = self.readWords(trainPath) # Read words from training set (remove punctuation)
        self.validWords = self.readWords(validPath) # Read words from validation set (remove punctuation)
        self.testWords = self.readWords(testPath) # Read words from test set (remove punctuation)
        print("Extracting dictionary...")
        self.trainDict, self.trainUnkDict = self.extractDict(self.trainWords) # Extract dictionary from training set and mark unknown words (1 frequency)
        self.validDict, self.validUnkDict = self.extractValTestDict(self.validWords) # Extract dictionary from validation set and mark unknown words (1 frequency)
        self.testDict, self.testUnkDict = self.extractValTestDict(self.testWords) # Extract dictionary from test set and mark unknown words (1 frequency)
        print("Extracting ngrams...")
        self.trainTwoGrams = self.extractNgrams(self.trainWords, context_size, self.trainUnkDict) # Extract ngrams from training set and mark unknown words (1 frequency)
        self.validTwoGrams = self.extractNgrams(self.validWords, context_size, self.validUnkDict) # Extract ngrams from validation set and mark unknown words (1 frequency)
        self.testTwoGrams = self.extractNgrams(self.testWords, context_size, self.testUnkDict) # Extract ngrams from test set and mark unknown words (1 frequency)
        print("Done preprocessing!")
    
    '''Read in the corpus and return a list of words. 
    Punctuation is removed.'''
    def readWords(self, path):
        file = open(path, "r")
        lines = " ".join(file.read().translate(str.maketrans('', '', string.punctuation)).splitlines()) # Remove punctuation 
        words = lines.split(" ") # Split into words
        return words

    '''Extract a frequency dictionary and mark unknown words 
    Returns a dictionary and a dictionary of unknown word booleans
    This function is used for validation and test corpi only where UNKs are defined by not being in the training corpus'''
    def extractValTestDict(self, words):
        wordDict = {} # Initialize dictionary
        unkDict = {} # Initialize dictionary of unknown words
        unkCount = 0 # Initialize unknown word count

        #Extract dict of word frequency's from corpus
        for word in words:
            if word in wordDict:
                wordDict[word] += 1 # Increment frequency of word
            else:
                wordDict[word] = 1 # Set frequency of word to 1
        #Extract dict of unknown words from corpus
        for word in list(wordDict.keys()):
            if word not in self.trainUnkDict or self.trainUnkDict[word] == False:
                del wordDict[word] # Delete word from dictionary if it is not in the training corpus or freq = 1
                unkDict[word] = False # Mark word as unknown
                unkCount += 1 # Increment unknown word count
            else:
                unkDict[word] = True # Mark word as known
        wordDict["<UNK>"] = unkCount # Add unknown word to dictionary with frequency = unknown word count
        return wordDict, unkDict

    '''Extract a frequency dictionary and mark unknown words (1 frequency)
    Returns a dictionary and a dictionary of unknown word booleans
    This function is used for training corpus only where UNKs are defined by occurring once in the training corpus'''
    def extractDict(self, words):
        wordDict = {} # Initialize dictionary
        unkDict = {} # Initialize dictionary of unknown words

        #Extract dict of word frequency's from corpus
        for word in words:
            if word in wordDict:
                wordDict[word] += 1 # Increment frequency of word
            else:
                wordDict[word] = 1 # Set frequency of word to 1
        unkCount = 0 # Initialize unknown word count
        for word in list(wordDict.keys()):
            if wordDict[word] == 1:
                del wordDict[word] # Delete word from dictionary if it is only 1 frequency
                unkDict[word] = False # Mark word as unknown
                unkCount += 1 # Increment unknown word count
            else:
                unkDict[word] = True # Mark word as known
        wordDict["<UNK>"] = unkCount # Add unknown word to dictionary with frequency = unknown word count
        return wordDict, unkDict


    '''Extract a list of ngrams from a list of words
    Words replaced by <UNK> if marked as unknown'''
    def extractNgrams(self, words, n, unkDict):
        ngrams = [(
                    [words[i - j - 1] if unkDict[words[i-j-1]] else "<UNK>" for j in range(n)],
                    words[i] if unkDict[words[i]] else "<UNK>"
                )
                for i in range(n, len(words))
                ]
        return ngrams

    '''Extract a list of ngrams from a list of words no <UNK> replacement
    This function was just for testing if UNK replacement worked'''
    def extractNgramsNoUnk(self, words, n):
        ngrams = [(
                    [words[i - j - 1] for j in range(n)],
                    words[i]
                )
                for i in range(n, len(words))
                ]
        return ngrams

    # Get training freq dictionary
    def getTrainDict(self):
        return self.trainDict
    
    # Get validation freq dictionary
    def getValidDict(self):
        return self.validDict

    # Get test freq dictionary
    def getTestDict(self):
        return self.testDict
    
    # Get train two grams
    def getTrainTwoGrams(self):
        return self.trainTwoGrams

    # Get validation two grams
    def getValidTwoGrams(self):
        return self.validTwoGrams

    def getTestTwoGrams(self):
        return self.testTwoGrams