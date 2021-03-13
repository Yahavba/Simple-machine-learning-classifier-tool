import csv


class Loader:
    """
    class to load data form csv file, create test and training sets of data for data mining process
    """
    def __init__(self):
        """"
        Ctor for DataLoader
        """
        self.testSet = []
        self.trainingSet = []
        self.structure = []

    def loadData(self, pathOfFile):
        """
        method to read data csv file and build data structure, training data set and test data set. data is saved in class parameters
        Attributes:
            pathOfFile(string): the path to the data set csv file
        Raise:
            EnvironmentError
        """
        with open(pathOfFile) as csv_file:
            lines, csv_reader = [], csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                lines += [row]
        if len(lines) == 0 or len(lines[0]) <= 1:
            raise EnvironmentError
        self.buildStructure(lines)
        self.buildDataSets(lines[1:], self.structure)

    def buildStructure(self, lines):
        """"
        method to build structure ( column and their values) of data set
        Returns:
            structure(dict): the structure of data set returns {} if data set is empty, each element is
            columnName : {'index': index , 'values': [values]} or
            columnName : {'index': index , 'values': ["Numeric"]}
        """
        self.structure = self.getColumnsName(lines)
        for column in self.structure.keys():
            self.structure[column]['values'] = self.getColumnValues(list(self.structure.keys()).index(column), lines,
                                                                    self.structure['class']['index'])

    def getColumnsName(self, lines):
        """"
        method to get columns names from data set
        Attributes:
            lines (list): the lines in data set
        Returns:
            names (list): the names of columns in data set
        """
        names = {}
        for name in lines[0]:
            names[name] = {'index': lines[0].index(name)}
        return names

    def getColumnValues(self, columnIndex, lines, classIndex):
        """"
        method to get column values from data set
        Attributes:
            lines (list): the lines in data set
            columnIndex (int) : the index of the column to find his values
        Returns:
            values (list): the values of a column in data set
        """
        values = ["Numeric"]
        if not self.isNumeric(columnIndex, lines) or columnIndex == classIndex:
            values = []
            for line in lines[1:]:
                if line[columnIndex] != "" and values.count(line[columnIndex]) == 0:
                    values.append(line[columnIndex])
        return values

    def isNumeric(self, columnIndex, lines):
        """"
        method to check if column is Numeric type of data
        Attributes:
            lines (list): the lines in data set
            columnIndex (int) : the index of the column to find his values
        Returns:
            (boolean) : True if numeric else False
        """
        for line in lines[1:]:
            if line[columnIndex] != "":
                try:
                    int(line[columnIndex])
                except ValueError:
                    try:
                        float(line[columnIndex])
                    except ValueError:
                        return False
        return True

    def buildDataSets(self, lines, structure):
        """
        method to create training data set and test data set from data
        Attributes:
            lines(list): the lines in data set without first line of column names
            structure(dict): the structure of data
        """
        classIndex = structure['class']['index']
        for classValue in structure['class']['values']:
            data = list(filter(lambda x: x[classIndex] == classValue, lines))
            self.trainingSet += data[0:int(((len(data)*2)/3) + 0.5)]
            self.testSet += data[int(((len(data) * 2) / 3) + 0.5):]
        data = list(filter(lambda x: x[classIndex] == "", lines))
        self.trainingSet += data[0:int(((len(data) * 2) / 3) + 0.5)]
        self.testSet += data[int(((len(data) * 2) / 3) + 0.5):]

