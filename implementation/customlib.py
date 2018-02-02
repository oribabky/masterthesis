#Filter out certain indices in the inner array of a 2-d array
def sliceSkip2d(list, skips):
    resultArray = []

    for i in list:
        innerArray = []
        for k in range(0, len(i)):
            if k in skips:
                continue
            innerArray.append(i[k])
        resultArray.append(innerArray)
    return resultArray

def splitData(percentageTraining, inputList):
    resultArray = []

    size = len(inputList)
    nrTrain = int( round(percentageTraining*size/100) )
    trainArray = inputList[:nrTrain]
    testArray = inputList[nrTrain:]

    resultArray.append(trainArray)
    resultArray.append(testArray)
    return resultArray

#print(splitData(50, [1,2,3,4,5,6,7,8,9]))
#print(sliceSkip2d([[1,2,3],[4,5,6],[7,8,9]], [1]))