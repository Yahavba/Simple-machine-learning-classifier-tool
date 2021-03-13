from MiningCalculations import MiningCalculator


class Discretization:
    def __init__(self):
        """"
        Ctor for Discretization
        """
        self.miningCalculator = MiningCalculator()

    def discretizationData(self, trainData, testData, structure, numOfBins, typeOfDiscretization):
        """
        method to apply discretization on each numeric column in data
        Attributes:
            trainData(list) : list of lines in test data set each element is a list
            testData(list) : list of lines in test data set each element is a list
            structure(dict): the structure of data set returns {} if data set is empty, each element is
                            columnName : {'index': index , 'values': [values]} or
                            columnName : {'index': index , 'values': ["Numeric"]
            numOfBins(int): number of bins for discretization
            typeOfDiscretization(string): what method of discretization if input does not fit a method entropy based discretization
            will be applied
        """
        for columnName, value in structure.items():
            if value["values"] == ['Numeric']:
                colIndex = value['index']
                bins = []
                self.sortDataByAscendingOrderOFValuesInColumn(trainData, colIndex)
                if typeOfDiscretization.upper() == "EQUAL WIDTH":
                    bins = self.createBinsByEqualWidth(trainData, colIndex, numOfBins)
                elif typeOfDiscretization.upper() == "EQUAL DEPTH":
                    bins = self.createBinsByEqualDepth(trainData, colIndex, numOfBins)
                elif typeOfDiscretization.upper() == "GINI INDEX":
                    bins = self.createBinsByGiniIndex(trainData, structure, colIndex, numOfBins)
                elif typeOfDiscretization.upper() == "ENTROPY":
                    bins = self.createBinsByEntropy(trainData, structure, columnName, numOfBins)
                else:
                    bins = self.createBinsByEntropy(trainData, structure, columnName, numOfBins)
                self.discretizationOFDataByColumn(trainData, colIndex, bins)
                self.discretizationOFDataByColumn(testData, colIndex, bins)
                structure[columnName]['values'] = list(bins.keys())

    def discretizationOFDataByColumn(self, data, colIndex, bins):
        """
        method to apply discretization on column in data
        Attributes:
            data(list) : list of lines in data set each element is a list
            colIndex(int): the index of column to apply discretization
            bins(dict): a dict with string representations of the bin and its value is a function to check if value belongs to bin
            example - {"value<X" : lambda x: x<x...}
        """
        for row in data:
            for bin, checkBinFunc in bins.items():
                if checkBinFunc(float(row[colIndex])):
                    row[colIndex] = bin
                    break

    def sortDataByAscendingOrderOFValuesInColumn(self, data, colIndex):
        """
        method to sort data by ascending values in column
        Attributes:
            data(list) : list of lines in data set each element is a list
            colIndex(int): the index of column to apply discretization
        """
        def sortFunc(x):
            return float(x[colIndex])
        data.sort(key=sortFunc)

    def createBinsByEqualWidth(self, data, colIndex, numOfBins):
        """
        method to create a bins dict by Equal Width technique each key is a string representations of the bin and its value is a function
        to check if some value belongs to the bin example {"value<X" : lambda x: x<x...}
        Attributes:
            data(list) : list of lines in data set each element is a list
            colIndex(int): the index of column to create bins from
            numOfBins(int): number of bins to create
        Returns:
            dict: bins dict by Equal Width technique each key is a string representations of the bin and its value is a function to check
            if some value belongs to the bin example {"value<X" : lambda x: x<x...}
            Attributes:
        """
        colData = list(map(lambda x: float(x[colIndex]), data))
        minVal, maxVal = min(colData), max(colData)
        width = round(((maxVal - minVal) / numOfBins), 3)
        bins = {"value<="+str(width): lambda x: x <= width}
        for i in range(1, numOfBins-1):
            bins[str(width) + '<value<=' + str(width+width)] = (lambda x:  width < x <= width + width)
            width = width + width
        bins["value>" + str(width)] = (lambda x: x > width)
        return bins

    def createBinsByEqualDepth(self, data, colIndex, numOfBins):
        """
        method to create a bins dict by Equal Depth technique each key is a string representations of the bin and its value is a function
        to check if some value belongs to the bin example {"value<X" : lambda x: x<x...}
        Attributes:
            data(list) : list of lines in data set each element is a list
            colIndex(int): the index of column to create bins from
            numOfBins(int): number of bins to create
        Returns:
            dict: bins dict by Equal Depth technique each key is a string representations of the bin and its value is a function to check
            if some value belongs to the bin example {"value<X" : lambda x: x<x...}
            Attributes:
        """
        colData = list(map(lambda x: float(x[colIndex]), data))
        Depth, splittedData, index = int(((len(colData) / numOfBins) + 1)), [], 0
        for i in range(0, numOfBins):
            splittedData, index, Depth = splittedData + [colData[index:Depth]], Depth, Depth + Depth
        bins = {"value<=" + str(max(splittedData[0])): lambda x: x <= max(splittedData[0])}
        index = 1
        while index < numOfBins-1:
            if max(splittedData[index-1]) != max(splittedData[index]):
                val1 = max(splittedData[index-1])
                val2 = max(splittedData[index])
                bins[str(val1) + '<value<=' + str(val2)] = (lambda x: val1 < x <= val2)
            index += 1
        bins["value>" + str(max(splittedData[index-1]))] = (lambda x: x > max(splittedData[index-1]))
        return bins

    def createBinsByEntropy(self, data, structure, colName, numOfBins):
        """
        method to create a bins dict by Entropy technique each key is a string representations of the bin and its value is a function
        to check if some value belongs to the bin example {"value<X" : lambda x: x<x...}
        Attributes:
            data(list) : list of lines in data set each element is a list
            structure(dict): the structure of data set returns {} if data set is empty, each element is
                            columnName : {'index': index , 'values': [values]} or
                            columnName : {'index': index , 'values': ["Numeric"]
            colName(int): the name of column to create bins from
            numOfBins(int): number of bins to create
        Returns:
            dict: bins dict by Entropy technique each key is a string representations of the bin and its value is a function to check
            if some value belongs to the bin example {"value<X" : lambda x: x<x...}
            Attributes:
        """
        splits = self.miningCalculator.getBestSplitsInDataByInfoGain(data, structure, colName, numOfBins-1)
        splits.sort()
        bins = {"value<="+str(splits[0]): lambda x: x <= splits[0]}
        if len(splits) > 1:
            for i in range(1, numOfBins-1):
                bins[str(splits[i-1]) + '<value<=' + str(splits[i])] = (lambda x:  splits[i-1] < x <= splits[i])
        bins["value>" + str(splits[len(splits)-1])] = (lambda x: x > splits[len(splits)-1])
        return bins

    def createBinsByGiniIndex(self, data, structure, colIndex, numOfBins):
        """
        method to create a bins dict by Gini Index technique each key is a string representations of the bin and its value is a function
        to check if some value belongs to the bin example {"value<X" : lambda x: x<x...}
        Attributes:
            data(list) : list of lines in data set each element is a list
            colIndex(int): the index of column to create bins from
            numOfBins(int): number of bins to create
        Returns:
            dict: bins dict by Gini Index technique each key is a string representations of the bin and its value is a function to check
            if some value belongs to the bin example {"value<X" : lambda x: x<x...}
            Attributes:
        """
        splits = self.miningCalculator.getListWithBestValueSplitsOfDataByGini(data, structure, colIndex, numOfBins - 1)
        splits.sort()
        bins = {"value<=" + str(splits[0]): lambda x: x <= splits[0]}
        if len(splits) > 1:
            for i in range(1, numOfBins - 1):
                bins[str(splits[i - 1]) + '<value<=' + str(splits[i])] = (lambda x: splits[i - 1] < x <= splits[i])
        bins["value>" + str(splits[len(splits) - 1])] = (lambda x: x > splits[len(splits) - 1])
        return bins




