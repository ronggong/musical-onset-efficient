# -*- coding: utf-8 -*-

import csv
import pinyin
import os
from zhon.hanzi import punctuation as puncChinese
import re

def csvDurationScoreParser(scoreFilename):

    syllables = []
    syllable_durations = []
    bpm                 = []

    with open(scoreFilename, 'rb') as csvfile:
        score = csv.reader(csvfile)
        for idx, row in enumerate(score):
            if (idx+1)%2:
                syllables.append(row[1:])
            if idx%2:
                syllable_durations.append(row[1:])
                bpm.append(row[0])

    return syllables, syllable_durations, bpm

def csvScorePinyinParser(scoreFilename):
    syllables = []
    syllable_durations = []
    bpm = []
    pinyins = []

    with open(scoreFilename, 'rb') as csvfile:
        score = csv.reader(csvfile)
        for idx, row in enumerate(score):
            if idx%3 == 0:
                syllables.append(row[1:])
            if idx%3 == 1:
                pinyins.append(row[1:])
            if idx%3 == 2:
                syllable_durations.append(row[1:])
                bpm.append(row[0])
    return syllables, pinyins, syllable_durations, bpm

def removePunctuation(char):
    if len(re.findall(ur'[\u4e00-\u9fff]+', char.decode("utf8"))):
        char = re.sub(ur"[%s]+" % puncChinese, "", char.decode("utf8"))
    return char

def generatePinyin(scoreFilename):

    syllables           = []
    pinyins             = []
    syllable_durations  = []
    bpm                 = []
    try:
        with open(scoreFilename, 'rb') as csvfile:
            score = csv.reader(csvfile)
            for idx, row in enumerate(score):
                if (idx+1)%2:
                    syllables.append(row[1:])

                    row_pinyin = []
                    for r in row[1:]:
                        if len(r):
                            r = removePunctuation(r)
                            row_pinyin.append(pinyin.get(r, format="strip", delimiter=" "))
                        else:
                            row_pinyin.append('')
                    pinyins.append(row_pinyin)

                elif idx%2:
                    syllable_durations.append(row[1:])
                    bpm.append(row[0])
    except IOError:
        print scoreFilename, 'not found.'

    return syllables,pinyins,syllable_durations,bpm

def writerowCsv(syllables,syllable_durations,bpm,writer,pinyins=None):
    for ii in range(len(syllables)):
        writer.writerow(['']+syllables[ii])
        if pinyins:
            writer.writerow(['']+pinyins[ii])
        writer.writerow([bpm[ii]]+syllable_durations[ii])

def writeCsv(scoreFilename, syllables, syllable_durations, bpm):
    """
    write csv scores
    :param scoreFilename:
    :param syllables:
    :param syllable_durations:
    :param bpm:
    :return:
    """
    if len(syllables):

        directory, _ = os.path.split(scoreFilename)
        if not os.path.exists(directory):
            os.makedirs(directory)

        export=open(scoreFilename, "wb")
        writer=csv.writer(export, delimiter=',')
        writerowCsv(syllables,syllable_durations,bpm,writer,None)
        export.close()

def writeCsvPinyinFromData(scoreFilenamePinyin, syllables, pinyins, syllable_durations, bpm):
    if len(syllables):
        export=open(scoreFilenamePinyin, "wb")
        writer=csv.writer(export, delimiter=',')
        writerowCsv(syllables,syllable_durations,bpm,writer,pinyins)
        export.close()

def writeCsvPinyin(scoreFilename, scoreFilenamePinyin):
    '''
    use this function to add pinyin to scoreFilename
    :param scoreFilename:
    :param scoreFilenamePinyin:
    :return:
    '''
    syllables,pinyins,syllable_durations,bpm = generatePinyin(scoreFilename)
    writeCsvPinyinFromData(scoreFilenamePinyin, syllables, pinyins, syllable_durations, bpm)
