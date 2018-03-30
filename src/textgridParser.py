# -*- coding: utf-8 -*-
import sys, os

currentPath = os.path.dirname(__file__)
utilsPath = os.path.join(currentPath, 'utils')
sys.path.append(utilsPath)

import textgrid as tgp

def textGrid2WordList(textgrid_file, whichTier = 'pinyin', utf16 = True):
    '''
    parse textGrid into a python list of tokens 
    @param whichTier : 'pinyin' default tier name  
    '''	
    if not os.path.isfile(textgrid_file): raise Exception("file {} not found".format(textgrid_file))
    beginTsAndWordList = []

    if utf16:
        par_obj = tgp.TextGrid.loadUTF16(textgrid_file)	#loading the object
    else:
        par_obj = tgp.TextGrid.load(textgrid_file)	#loading the object

    tiers= tgp.TextGrid._find_tiers(par_obj)	#finding existing tiers		
	
    isTierFound = False
    for tier in tiers:
        tierName= tier.tier_name().replace('.','')
        #iterating over tiers and selecting the one specified
        if tierName == whichTier:
            isTierFound = True
            #this function parse the file nicely and return cool tuples
            tier_details = tier.make_simple_transcript()

            for line in tier_details:
                beginTsAndWordList.append([float(line[0]), float(line[1]), line[2]])

    if not isTierFound:
        print ('Missing tier {1} in file {0}' .format(textgrid_file, whichTier))

    return beginTsAndWordList

def line2WordList(line, entireWordList):
    '''
    find the nested wordList of entireWordList by line tuple
    :param line: line tuple [startTime, endTime, string]
    :param entireWordList: entire word list
    :return: nested wordList
    '''
    nestedWordList = []
    vault = False
    for wordlist in entireWordList:
         # the ending of the line
        if wordlist[1] == line[1]:
            nestedWordList.append(wordlist)
            break
        # the beginning of the line
        if wordlist[0] == line[0]:
            vault = True
        if vault == True:
            nestedWordList.append(wordlist)

    return nestedWordList

def wordListsParseByLines(entireLine, entireWordList):
    '''
    find the wordList for each line, cut the word list according to line
    :param entireLine: entire lines in line tier
    :param entirewWordList: entire word lists in pinyin tier
    :return:
    nestedWordLists: [[line0, wordList0], [line1, wordList1], ...]
    numLines: sum of number of lines
    numWords: sum of number of words
    '''
    nestedWordLists     = []
    numLines            = 0
    numWords            = 0

    for line in entireLine:
        asciiLine=line[2].encode("ascii", "replace")
        if len(asciiLine.replace(" ", "")):                                      # if line is not empty
            numLines        += 1
            nestedWordList  = []
            wordList        = line2WordList(line, entireWordList)
            for word in wordList:
                asciiWord = word[2].encode("ascii", "replace")
                if len(asciiWord.replace(" ","")):                              # if word is not empty
                    numWords += 1
                    nestedWordList.append(word)
            nestedWordLists.append([line,nestedWordList])

    return nestedWordLists, numLines, numWords

def syllableTextgridExtraction(textgrid_path, recording, tier0, tier1):

    '''
    Extract syllable boundary and phoneme boundary from textgrid
    :param textgrid_path:
    :param recording:
    :param tier0: parent tier
    :param tier1: child tier which should be covered by parent tier
    :return:
    nestedPhonemeList, element[0] - syllable, element[1] - a list containing the phoneme of the syllable
    '''

    textgrid_file   = os.path.join(textgrid_path,recording+'.TextGrid')

    syllableList    = textGrid2WordList(textgrid_file, whichTier=tier0)
    phonemeList     = textGrid2WordList(textgrid_file, whichTier=tier1)

    # parse syllables of groundtruth
    nestedPhonemeLists, numSyllables, numPhonemes   = wordListsParseByLines(syllableList, phonemeList)

    return nestedPhonemeLists, numSyllables, numPhonemes


