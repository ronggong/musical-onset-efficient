
def syllables_total_parser(filename):
    '''
    parse Georgi's syllable alignment result
    :param filename:
    :return:
    '''

    with open(filename) as file:	# Use file to refer to the file object
        boundaryList = []
        lines = file.readlines()
        for idx,line in enumerate(lines):
            if not idx == 0:
                lineWoLB    = line.rstrip('\n')
                lineList    = lineWoLB.split("\t")
                if not lineList[2].strip() == 'REST':
                    lineList    = [float(lineList[0].strip()), float(lineList[1].strip()), lineList[2].strip()]
                    boundaryList.append(lineList)

    return boundaryList