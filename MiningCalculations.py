from math import log2


class MiningCalculator:
    def __init__(self):
        pass

    # Entropy calculations
    def calcDataEntropy(self, data, structure):
        """
        method to calculate entropy of class in data set
        Attributes:
            data(list) : list of lines in files each element is a list
            structure(dict): the structure of data set returns {} if data set is empty, each element is
                            columnName : {'index': index , 'values': [values]} or
                            columnName : {'index': index , 'values': ["Numeric"]
        Returns:
            float: the entropy of class in data set
        """
        classIndex, entropy = structure['class']['index'], 0
        for value in structure['class']['values']:
            newData = list(filter(lambda y: y[classIndex] == value, data))
            p = len(newData) / len(data) if len(data) > 0 else 1
            entropy += (-1) * (p * log2(p)) if p > 0 else 0
        return round(entropy, 3)

    def calcEntropyBySplitValue(self, data, structure, colName, splitVal):
        """
        method to calculate entropy of class in data set if we split it by a numeric split value
        Attributes:
            data(list) : list of lines in files each element is a list
            structure(dict): the structure of data set returns {} if data set is empty, each element is
                            columnName : {'index': index , 'values': [values]} or
                            columnName : {'index': index , 'values': ["Numeric"]
            colName(String): the name of column to split data
            splitVal(float): the number to split data
        Returns:
            float: the entropy of splitting data by numeric split value
        """
        colIndex, entropy = structure[colName]['index'], 0
        newDataBellowSplit = list(filter(lambda y: float(y[colIndex]) <= splitVal, data))
        newDataAboveSplit = list(filter(lambda y: float(y[colIndex]) > splitVal, data))
        entropyOfNewDataBellowSplit = self.calcDataEntropy(newDataBellowSplit, structure)
        entropyOfNewAboveSplit = self.calcDataEntropy(newDataAboveSplit, structure)
        entropy += (len(newDataBellowSplit) / len(data)) * entropyOfNewDataBellowSplit
        entropy += (len(newDataAboveSplit) / len(data)) * entropyOfNewAboveSplit
        return round(entropy, 3)

    def calcInfoGainBySplitValue(self, data, structure, colName, splitVal):
        """
        method to calculate info-gain of splitting data by a a
        Attributes:
            data(list) : list of lines in files each element is a list
            structure(dict): the structure of data set returns {} if data set is empty, each element is
                            columnName : {'index': index , 'values': [values]} or
                            columnName : {'index': index , 'values': ["Numeric"]
            colName(String): the name of column to split data
        Returns:
            float: the info-gain of class in data set after splitting data by column
        """
        result = self.calcDataEntropy(data, structure) - self.calcEntropyBySplitValue(data, structure, colName, splitVal)
        result = 0 if result < 0 else result
        return round(result, 3)

    def findBestSplitInDataByInfoGain(self, data, structure, colName):
        """
        method to find best split in the data by info-gain
        Attributes:
            data(list) : list of lines in files each element is a list
            structure(dict): the structure of data set returns {} if data set is empty, each element is
                            columnName : {'index': index , 'values': [values]} or
                            columnName : {'index': index , 'values': ["Numeric"]
            colName(String): the name of column to find splits of data
        Returns:
            list: best split in the data by info-gain and its info gain value example - [split, infoGain]
        """
        colIndex, maxInfoGain, bestSplit = structure[colName]['index'], 0, []
        for i in range(0, len(data)-1):
            split = (float(data[i][colIndex]) + float(data[i+1][colIndex])) / 2
            infoGain = self.calcInfoGainBySplitValue(data, structure, colName, split)
            if infoGain >= maxInfoGain:
                bestSplit = [split, infoGain]
                maxInfoGain = infoGain
        return bestSplit

    def calcEntropyByColumnSplit(self, data, structure, colName):
        """
        method to calculate entropy of class in data set if we split it by column
        Attributes:
            data(list) : list of lines in files each element is a list
            structure(dict): the structure of data set returns {} if data set is empty, each element is
                            columnName : {'index': index , 'values': [values]} or
                            columnName : {'index': index , 'values': ["Numeric"]
            colName(String): the name of column to split data
        Returns:
            float: the entropy of splitting data by column
        """
        colIndex, entropy = structure[colName]['index'], 0
        for colValue in structure[colName]['values']:
            newData = list(filter(lambda y: y[colIndex] == colValue, data))
            entropyOfNewData = self.calcDataEntropy(newData, structure)
            entropy += (len(newData)/len(data)) * entropyOfNewData
        return round(entropy, 3)

    def calcInfoGainByColumnSplit(self, data, structure, colName):
        """
        method to calculate info-gain of splitting data set by numeric split value
        Attributes:
            data(list) : list of lines in files each element is a list
            structure(dict): the structure of data set returns {} if data set is empty, each element is
                            columnName : {'index': index , 'values': [values]} or
                            columnName : {'index': index , 'values': ["Numeric"]
            colName(String): the name of column to split data
        Returns:
            float: the info-gain of class in data set after splitting data by column
        """
        result = self.calcDataEntropy(data, structure) - self.calcEntropyByColumnSplit(data, structure, colName)
        result = 0 if result < 0 else result
        return round(result, 3)

    def findBestColumnSplitByInfoGain(self, data, structure):
        """
        method to find best column to split data by Info Gain
        Attributes:
            data(list) : list of lines in files each element is a list
            structure(dict): the structure of data set returns {} if data set is empty, each element is
                            columnName : {'index': index , 'values': [values]} or
                            columnName : {'index': index , 'values': ["Numeric"]
        Returns:
            String: the column best to split data by Info Gain
        """
        maxInfoGain, bestSplit = 0, None
        for colName in list(structure.keys())[:-1]:
            infoGain = self.calcInfoGainByColumnSplit(data, structure, colName)
            if infoGain >= maxInfoGain:
                maxInfoGain = infoGain
                bestSplit = colName
        return bestSplit

    def fillBestSplitsInDataByInfoGainIntoDict(self, data, structure, colName, numOfSplits, splitsList, indexToInsert):
        """
        recursive method to fill a list with best splits in the data by info-gain
        Attributes:
            data(list) : list of lines in files each element is a list
            structure(dict): the structure of data set returns {} if data set is empty, each element is
                            columnName : {'index': index , 'values': [values]} or
                            columnName : {'index': index , 'values': ["Numeric"]
            colName(String): the name of column to find splits of data
            numOfSplits(int): number of splits to find
            splitsList(list): list to fill splits in it
            indexToInsert(int): an index to insert the splits in dict to keep splits order in recursive method
        Returns:
            dict: dict of number of split (when it Happens) and value of list of
            best split in the data by info-gain and its info gain value example - {split number: [[split, infoGain],[split, infoGain]]
        """
        if len(data) <= 0 or numOfSplits <= 0:
            return []
        colIndex = structure[colName]['index']
        split = self.findBestSplitInDataByInfoGain(data, structure, colName)
        if str(indexToInsert) in splitsList:
            splitsList[str(indexToInsert)] += [split]
        else:
            splitsList[str(indexToInsert)] = [split]
        indexToInsert, numOfSplits = indexToInsert + 1, numOfSplits - 1

        if split:
            newDataBellowSplit = list(filter(lambda y: float(y[colIndex]) <= split[0], data))
            newDataAboveSplit = list(filter(lambda y: float(y[colIndex]) > split[0], data))
            self.fillBestSplitsInDataByInfoGainIntoDict(newDataBellowSplit, structure, colName, numOfSplits, splitsList, indexToInsert)
            self.fillBestSplitsInDataByInfoGainIntoDict(newDataAboveSplit, structure, colName, numOfSplits, splitsList, indexToInsert)

    def getBestSplitsInDataByInfoGain(self, data, structure, colName, numOfSplits):
        """
        method to get a list with best splits in the data by info-gain
        Attributes:
            data(list) : list of lines in files each element is a list
            structure(dict): the structure of data set returns {} if data set is empty, each element is
                            columnName : {'index': index , 'values': [values]} or
                            columnName : {'index': index , 'values': ["Numeric"]
            colName(String): the name of column to find splits of data
            numOfSplits(int): number of splits to find
        Returns:
            list: best splits in the data by info-gain ordered by best split to take, example - [splitOne, splitTwo, SplitThree...]
        """
        splitsList, newSplitsList = {}, []
        self.fillBestSplitsInDataByInfoGainIntoDict(data, structure, colName, numOfSplits, splitsList, 0)
        for lists in list(splitsList.values())[1:]:
            while len(lists) > 0:
                splitOne, splitTwo = [], []
                if len(lists) > 0:
                    splitOne = lists.pop()
                if len(lists) > 0:
                    splitTwo = lists.pop()
                if splitOne and splitTwo:
                    if splitOne[1] >= splitTwo[1]:
                        newSplitsList.append(splitOne[0])
                        newSplitsList.append(splitTwo[0])
                    else:
                        newSplitsList.append(splitTwo[0])
                        newSplitsList.append(splitOne[0])
                elif splitOne:
                    newSplitsList.append(splitOne[0])
                elif splitTwo:
                    newSplitsList.append(splitTwo[0])

        newSplitsList.insert(0, splitsList['0'][0][0])
        self.removeDuplicatesInList(newSplitsList)
        return newSplitsList[0:numOfSplits]

    # Gini calculations
    def calcDataGini(self, data, structure):
        """
        method to calculate gini of class in data set
        Attributes:
            data(list) : list of lines in files each element is a list
            structure(dict): the structure of data set returns {} if data set is empty, each element is
                            columnName : {'index': index , 'values': [values]} or
                            columnName : {'index': index , 'values': ["Numeric"]
        Returns:
            float: the gini of class in data set
        """
        classIndex, result, lenData = structure['class']['index'], 1, len(data)
        for value in structure['class']['values']:
            newData = list(filter(lambda x: x[classIndex] == value, data))
            p = len(newData) / lenData if lenData > 0 else 1
            result -= (p*p)
        return round(result, 3)

    def calcGiniSplitBySplitValue(self, data, structure, colIndex, splitValue):
        """
        method to calculate gini of class in data set if we split it by a numeric split value
        Attributes:
            data(list) : list of lines in files each element is a list
            structure(dict): the structure of data set returns {} if data set is empty, each element is
                            columnName : {'index': index , 'values': [values]} or
                            columnName : {'index': index , 'values': ["Numeric"]
            colIndex(String): the index of column to split data
            splitValue(float): the number to split data
        Returns:
            float: the gini of splitting data by numeric split value
        """
        dataBellow = list(filter(lambda x: float(x[colIndex]) <= splitValue, data))
        dataAbove = list(filter(lambda x: float(x[colIndex]) > splitValue, data))
        giniSplit = (len(dataBellow) / len(data)) * self.calcDataGini(dataBellow, structure) +\
                    (len(dataAbove) / len(data)) * self.calcDataGini(dataAbove, structure)
        return round(giniSplit, 3)

    def findBestValueSplitByGini(self, data, structure, colIndex):
        """
        method to find best split in the data by Gini
        Attributes:
            data(list) : list of lines in files each element is a list
            structure(dict): the structure of data set returns {} if data set is empty, each element is
                            columnName : {'index': index , 'values': [values]} or
                            columnName : {'index': index , 'values': ["Numeric"]
            colIndex(String): the index of column to find splits of data
        Returns:
            list: best split in the data by Gini and its Gini value example - [split, Gini]
        """
        minGini, bestSplit = 1, []
        for i in range(0, len(data)-1):
            split = (float(data[i][colIndex]) + float(data[i+1][colIndex])) / 2
            giniSplit = self.calcGiniSplitBySplitValue(data, structure, colIndex, split)
            if giniSplit <= minGini:
                minGini = giniSplit
                bestSplit = [split, giniSplit]
        return bestSplit

    def fillDictWithBestValueSplitsOfDataByGini(self, data, structure, colIndex, numOfSplits, splitsList, indexToInsert):
        """
        recursive method to fill a Dict with best splits in the data by Gini
        Attributes:
            data(list) : list of lines in files each element is a list
            structure(dict): the structure of data set returns {} if data set is empty, each element is
                            columnName : {'index': index , 'values': [values]} or
                            columnName : {'index': index , 'values': ["Numeric"]
            colIndex(String): the index of column to find splits of data
            numOfSplits(int): number of splits to find
            splitsList(list): list to fill splits in it
            indexToInsert(int): an index to insert the splits in dict to keep splits order in recursive method
        Returns:
            dict: dict of number of split (when it Happens) and value of list of
            best split in the data by Gini and its Gini value example - {split number: [[split, Gini],[split, Gini]]
        """
        if len(data) <= 0 or numOfSplits <= 0:
            return []
        split = self.findBestValueSplitByGini(data, structure, colIndex)
        if str(indexToInsert) in splitsList:
            splitsList[str(indexToInsert)] += [split]
        else:
            splitsList[str(indexToInsert)] = [split]
        indexToInsert, numOfSplits = indexToInsert + 1, numOfSplits - 1

        if split:
            newDataBellowSplit = list(filter(lambda y: float(y[colIndex]) <= split[0], data))
            newDataAboveSplit = list(filter(lambda y: float(y[colIndex]) > split[0], data))
            self.fillDictWithBestValueSplitsOfDataByGini(newDataBellowSplit, structure, colIndex, numOfSplits, splitsList, indexToInsert)
            self.fillDictWithBestValueSplitsOfDataByGini(newDataAboveSplit, structure, colIndex, numOfSplits, splitsList, indexToInsert)

    def getListWithBestValueSplitsOfDataByGini(self, data, structure, colIndex, numOfSplits):
        """
        method to get a list with best splits in the data by Gini
        Attributes:
            data(list) : list of lines in files each element is a list
            structure(dict): the structure of data set returns {} if data set is empty, each element is
                            columnName : {'index': index , 'values': [values]} or
                            columnName : {'index': index , 'values': ["Numeric"]
            colIndex(String): the index of column to find splits of data
            numOfSplits(int): number of splits to find
        Returns:
            list: best splits in the data by Gini ordered by best split to take, example - [splitOne, splitTwo, SplitThree...]
        """
        splitsList, newSplitsList = {}, []
        self.fillDictWithBestValueSplitsOfDataByGini(data, structure, colIndex, numOfSplits, splitsList, 0)
        for lists in list(splitsList.values())[1:]:
            while len(lists) > 0:
                splitOne, splitTwo = [], []
                if len(lists) > 0:
                    splitOne = lists.pop()
                if len(lists) > 0:
                    splitTwo = lists.pop()
                if splitOne and splitTwo:
                    if splitOne[1] <= splitTwo[1]:
                        newSplitsList.append(splitOne[0])
                        newSplitsList.append(splitTwo[0])
                    else:
                        newSplitsList.append(splitTwo[0])
                        newSplitsList.append(splitOne[0])
                elif splitOne:
                    newSplitsList.append(splitOne[0])
                elif splitTwo:
                    newSplitsList.append(splitTwo[0])

        newSplitsList.insert(0, splitsList['0'][0][0])
        self.removeDuplicatesInList(newSplitsList)
        return newSplitsList[0:numOfSplits]

    def calcGiniSplitByColumn(self, data, structure, colIName):
        """
        method to calculate gini of class in data set if we split it by a values in a column
        Attributes:
            data(list) : list of lines in files each element is a list
            structure(dict): the structure of data set returns {} if data set is empty, each element is
                            columnName : {'index': index , 'values': [values]} or
                            columnName : {'index': index , 'values': ["Numeric"]
            colIName(String): the name of column to split data
        Returns:
            float: the gini of splitting data by values in a column
        """
        colIndex, giniSplit = structure[colIName]['index'], 0
        for value in structure[colIName]["values"]:
            newData = list(filter(lambda x: x[colIndex] == value, data))
            p = len(newData) / len(data)
            giniSplit += self.calcDataGini(newData, structure) * p
        return round(giniSplit, 3)

    def findBestColumnSplitByGini(self, data, structure):
        """
        method to find best column to split data by gini
        Attributes:
            data(list) : list of lines in files each element is a list
            structure(dict): the structure of data set returns {} if data set is empty, each element is
                            columnName : {'index': index , 'values': [values]} or
                            columnName : {'index': index , 'values': ["Numeric"]
        Returns:
            String: the column best to split data by gini
        """
        minGini, bestSplit = 1, None
        for colName in list(structure.keys())[:-1]:
            giniSplit = self.calcGiniSplitByColumn(data, structure, colName)
            if giniSplit <= minGini:
                minGini = giniSplit
                bestSplit = colName
        return bestSplit

    # Gain Ratio calculations
    def calcGainRatioSplitByColumn(self, data, structure, colIName):
        """
        method to calculate GainRatio of class in data set if we split it by a values in a column
        Attributes:
            data(list) : list of lines in files each element is a list
            structure(dict): the structure of data set returns {} if data set is empty, each element is
                            columnName : {'index': index , 'values': [values]} or
                            columnName : {'index': index , 'values': ["Numeric"]
            colIName(String): the name of column to split data
        Returns:
            float: the GainRatio of splitting data by values in a column
        """
        splitInfo, colIndex = 0, structure[colIName]['index']
        for value in structure[colIName]['values']:
            newData = list(filter(lambda x: x[colIndex] == value, data))
            p = len(newData) / len(data) if len(newData) != 0 else 1
            splitInfo += (-1) * p * log2(p)
        splitInfo = 1 if splitInfo == 0 else splitInfo
        return round(self.calcInfoGainByColumnSplit(data, structure, colIName) / splitInfo, 3)

    def findBestColumnSplitByGainRatio(self, data, structure):
        """
        method to find best column to split data by GainRatio
        Attributes:
            data(list) : list of lines in files each element is a list
            structure(dict): the structure of data set returns {} if data set is empty, each element is
                            columnName : {'index': index , 'values': [values]} or
                            columnName : {'index': index , 'values': ["Numeric"]
        Returns:
            String: the column best to split data by GainRatio
        """
        maxGainRatio, bestSplit = 0, None
        for colName in list(structure.keys())[:-1]:
            GainRatio = self.calcGainRatioSplitByColumn(data, structure, colName)
            if GainRatio >= maxGainRatio:
                maxGainRatio = GainRatio
                bestSplit = colName
        return bestSplit

    def removeDuplicatesInList(self, data):
        """
        method to remove duplicates from list
        Attributes:
            data(list) : list of values
        Returns:
            list: data with no duplicates
        """
        newDataList = []
        for i in data:
            if newDataList.count(i) == 0:
                newDataList.append(i)
        data.clear()
        data += newDataList

    def mostCommonClassAttribute(self, data, structure):
        """
        method to find most common attribute in class column
        Attributes:
            data(list) : list of values
            structure(dict): the structure of data set returns {} if data set is empty, each element is
                            columnName : {'index': index , 'values': [values]} or
                            columnName : {'index': index , 'values': ["Numeric"]
        Returns:
            String: most common attribute in class column
        """
        maxCount, classIndex, mostCommonClassAttribute = 0, structure['class']['index'], None
        for value in structure['class']['values']:
            newData = list(filter(lambda y: y[classIndex] == value, data))
            if len(newData) >= maxCount:
                maxCount = len(newData)
                mostCommonClassAttribute = value
        return mostCommonClassAttribute

    def allRowsWithSameClass(self, data, structure):
        """
        method to check if all rows have the same class attribute
        Attributes:
            data(list) : list of values
            structure(dict): the structure of data set returns {} if data set is empty, each element is
                            columnName : {'index': index , 'values': [values]} or
                            columnName : {'index': index , 'values': ["Numeric"]
        Returns:
            Boolean: true if all rows have the same class attribute
        """
        classIndex = structure['class']['index']
        for value in structure['class']['values']:
            newData = list(filter(lambda x: x[classIndex] == value, data))
            if len(newData) == len(data):
                return True
        return False

    def calcNumberOfMajorityClassRows(self, data, structure):
        """
        method to calc number of rows with majority class value
        Attributes:
            data(list) : list of values
            structure(dict): the structure of data set returns {} if data set is empty, each element is
                            columnName : {'index': index , 'values': [values]} or
                            columnName : {'index': index , 'values': ["Numeric"]
        Returns:
            Boolean: number of rows with majority class value
        """
        maxCount, classIndex = 0, structure['class']['index']
        for value in structure['class']['values']:
            newData = list(filter(lambda y: y[classIndex] == value, data))
            if len(newData) >= maxCount:
                maxCount = len(newData)
        return maxCount

    def getSplitFunc(self, splitType):
        """
        method to get a column split function by string
        Attributes:
            splitType(list) : split function name
        Returns:
            function: split function by string
        """
        if splitType.upper() == "INFO GAIN":
            return self.findBestColumnSplitByInfoGain
        elif splitType.upper() == "GAIN RATIO":
            return self.findBestColumnSplitByGainRatio
        elif splitType.upper() == "GINI INDEX":
            return self.findBestColumnSplitByGini
        return None

    def calcProbabilityOfValGivenClassWithLaplaceCorrection(self, data, colIndex, val, classVal, numberOfValInColumn):
        """
        method calculate p(xi|ci) with laplace correction where xi is a value in column and ci is a class value
        Attributes:
            data(list) : list of rows in file
            colIndex(int) : the column index
            val(String): val of xi
            classVal(String): val of ci
            numberOfValInColumn(int): number of different values in column
        Returns:
            float: p(xi|xi) with laplace correction
        """
        newData = list(filter(lambda x: x[colIndex] == val and x[len(x)-1] == classVal, data))
        probability = (len(newData) + 1) / (len(data) + numberOfValInColumn) if len(data) > 0 else 0
        return round(probability, 3)

    def calcProbabilityOfClassValueWithLaplaceCorrection(self, data, classVal, numberOfClassValues):
        """
        method calculate p(ci) with laplace correction where ci is a class value
        Attributes:
            data(list) : list of rows in file
            classVal(String): val of ci
            numberOfClassValues(int): number of different class values
        Returns:
            float: p(ci) with laplace correction
        """
        newData = list(filter(lambda x: x[len(x)-1] == classVal, data))
        probability = (len(newData) + 1) / (len(data) + numberOfClassValues) if len(data) > 0 else 0
        return round(probability, 3)
