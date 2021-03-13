class Cleaner:
    def __init__(self):
        pass

    def cleanTrainingSet(self, data, structure):
        """
        method to clean training set as needed in postprocessing
        Attributes:
            data(list) : list of lines in files each element is a list
            structure(dict): the structure of data set returns {} if data set is empty, each element is
                            columnName : {'index': index , 'values': [values]} or
                            columnName : {'index': index , 'values': ["Numeric"]
        """
        self.removeRows(data, structure)
        self.fillMissingValues(data, structure)

    def cleanTestSet(self, data, structure):
        """
        method to clean test set as needed in postprocessing
        Attributes:
            data(list) : list of lines in files each element is a list
            structure(dict): the structure of data set returns {} if data set is empty, each element is
                            columnName : {'index': index , 'values': [values]} or
                            columnName : {'index': index , 'values': ["Numeric"]
        """
        self.fillMissingValues(data, structure)

    def removeRows(self, data, structure):
        """
        method to remove all rows with no class value
        Attributes:
            data(list) : list of lines in files each element is a list
            structure(dict): the structure of data set returns {} if data set is empty, each element is
                            columnName : {'index': index , 'values': [values]} or
                            columnName : {'index': index , 'values': ["Numeric"]
        """
        classIndex, linesToRemove = structure['class']['index'], []
        for line in data:
            if line[classIndex] == "":
                linesToRemove += [line]
        for line in linesToRemove:
            data.remove(line)

    def fillMissingValues(self, data, structure):
        """
        method to fill all types of missing values in data set rows
        Attributes:
            data(list) : list of lines in files each element is a list
            structure(dict): the structure of data set returns {} if data set is empty, each element is
                            columnName : {'index': index , 'values': [values]} or
                            columnName : {'index': index , 'values': ["Numeric"]
        """
        for column in structure.values():
            if column == structure["class"]:
                pass
            else:
                if str(column['values'][0]).upper() == "NUMERIC":
                    self.fillNumericValuesInColumn(data, structure, column['index'])
                else:
                    self.fillCategorialValuesInColumn(data, structure, column['index'])

    def fillNumericValuesInColumn(self, data, structure, indexOfCol):
        """
        method to fill numeric missing values in rows by the average value in column and class type or average
        value of column if there is no class type
        Attributes:
            data(list) : list of lines in files each element is a list
            structure(dict): the structure of data set returns {} if data set is empty, each element is
                            columnName : {'index': index , 'values': [values]} or
                            columnName : {'index': index , 'values': ["Numeric"]
            indexOfCol(int): the index of column we want to fill data
        """
        averages = self.AverageListByClass(data, structure, indexOfCol)
        newData = list(map(lambda x: float(x[indexOfCol]), filter(lambda y: y[indexOfCol] != "", data)))
        totalAverage = float(sum(newData)) / len(newData)
        totalAverage = str(round(totalAverage, 2))
        classIndex = structure['class']['index']
        for row in data:
            if row[indexOfCol] == "":
                if not row[classIndex]:
                    row[indexOfCol] = totalAverage
                else:
                    row[indexOfCol] = averages[(structure['class']['values']).index(row[classIndex])]
                    row[indexOfCol] = totalAverage if row[indexOfCol] is None else row[indexOfCol]

    def AverageListByClass(self, data, structure, indexOfCol):
        """
        method to get averages of values in column with a value of class attribute
        Attributes:
            data(list) : list of lines in files each element is a list
            structure(dict): the structure of data set returns {} if data set is empty, each element is
                            columnName : {'index': index , 'values': [values]} or
                            columnName : {'index': index , 'values': ["Numeric"]
            indexOfCol(int): the index of column we want to fill data
        Returns:
            list: averages of values in a column with a class value the order of the averages is in the same order of
            the class values in structure for example:
            values = [no,yes]
            averages = [10,20]
        """
        averages = []
        for value in structure['class']['values']:
            newData = list(filter(lambda x: x[structure['class']['index']] == value, data))
            columnData = list(map(lambda z: float(z), filter(lambda y: y != "", map(lambda x: x[indexOfCol], newData))))
            if len(columnData) > 0:
                average = float(sum(columnData)) / len(columnData)
                average = str(round(average, 2))
            else:
                average = None
            averages += [average]
        return averages

    def fillCategorialValuesInColumn(self, data, structure, indexOfCol):
        """
        method to fill categorical missing values in rows
        Attributes:
            data(list) : list of lines in files each element is a list
            structure(dict): the structure of data set returns {} if data set is empty, each element is
                            columnName : {'index': index , 'values': [values]} or
                            columnName : {'index': index , 'values': ["Numeric"]
            indexOfCol(int): the index of column we want to fill data
        """
        commonList = self.commonValuesByClass(data, structure, indexOfCol)
        mostCommon = self.mostFrequentElement(list(map(lambda x: x[indexOfCol], filter(lambda y: y[indexOfCol] != "", data))))
        classIndex = structure['class']['index']
        for row in data:
            if row[indexOfCol] == "":
                if row[classIndex] == "":
                    row[indexOfCol] = mostCommon
                else:
                    row[indexOfCol] = commonList[(structure['class']['values']).index(row[classIndex])]

    def commonValuesByClass(self, data, structure, indexOfCol):
        """
        method to get the most common value in a column for a class value
        Attributes:
            data(list) : list of lines in files each element is a list
            structure(dict): the structure of data set returns {} if data set is empty, each element is
                            columnName : {'index': index , 'values': [values]} or
                            columnName : {'index': index , 'values': ["Numeric"]
            indexOfCol(int): the index of column we want to fill data
        Returns:
            list: the most common value in a column for a class value, the order of the most commons is in the same
            order of the class values in structure for example:
            values = [no,yes]
            averages = [rich,poor]
        """
        common = []
        for value in structure['class']['values']:
            newData = list(filter(lambda x: x[structure['class']['index']] == value, data))
            columnData = list(filter(lambda y: y != "", map(lambda x: x[indexOfCol], newData)))
            common += [self.mostFrequentElement(columnData)]
        return common

    def mostFrequentElement(self, data):
        """
        method to get the most common value in a list
        Attributes:
            data(list) : list of lines in files each element is a list
        Returns:
            String: the most common value in a list
        """
        newData, mostFrequent, maxCount = [], "", 0
        for row in data:
            if newData.count(row) == 0:
                newData += row
                count = data.count(row)
                if count >= maxCount:
                    mostFrequent = row
                    maxCount = count
        return mostFrequent
