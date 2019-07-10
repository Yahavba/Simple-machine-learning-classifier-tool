import tkinter, csv, math, os
from tkinter import filedialog, messagebox
from functools import reduce


class Gui:
    """
    Class of id3 classifier gui
    """
    def __init__(self):
        """
         The constructor for Gui
        """
        self.folderPath = self.cleaned = self.discretizied = self.builded = self.classified = None
        self.root = tkinter.Tk()
        self.root.config(background="lightblue")
        self.root.resizable(False, False)
        self.root.title("Data mining")
        self.firstFrame = tkinter.Frame(self.root, bg="lightblue")
        self.firstFrame.pack(side=tkinter.TOP, fill=tkinter.X, padx=5, pady=5)
        self.folderPathLB = tkinter.Label(self.firstFrame, text="Folder path: ", height=1, width=23, bg="lightblue")
        self.folderPathLB.pack(side=tkinter.LEFT, fill=tkinter.X, padx=5, pady=5)
        self.folderPathTF = tkinter.Text(self.firstFrame, height=1, width=20)
        self.folderPathTF.pack(side=tkinter.LEFT, fill=tkinter.X, expand=1, padx=5, pady=5)
        self.folderPathBt = tkinter.Button(self.firstFrame, text="Browse", command=self.loadFilePathEvent)
        self.folderPathBt.pack(side=tkinter.LEFT, fill=tkinter.X, padx=5, pady=5)
        self.secondFrame = tkinter.Frame(self.root, bg="lightblue")
        self.secondFrame.pack(side=tkinter.TOP, fill=tkinter.X, padx=5, pady=5)
        self.discretizationLB = tkinter.Label(self.secondFrame, text="Discretization Bins: ", height=1, width=23, bg="lightblue")
        self.discretizationLB.pack(side=tkinter.LEFT, fill=tkinter.X, padx=5, pady=5)
        self.discretizationTF = tkinter.Text(self.secondFrame, height=1, width=23)
        self.discretizationTF.pack(side=tkinter.LEFT, fill=tkinter.X, expand=1, padx=5, pady=5)
        self.CleanBT = tkinter.Button(self.root, text="Clean", bg="white", width=33, command=self.cleanEvent)
        self.CleanBT.pack(side=tkinter.TOP, padx=5, pady=5)
        self.DiscretizationBT = tkinter.Button(self.root, text="Discretization", bg="white", width=33, command=self.discretizationEvent)
        self.DiscretizationBT.pack(side=tkinter.TOP, padx=5, pady=5)
        self.buildBT = tkinter.Button(self.root, text="Build", bg="white", width=33, command = self.buildEvent)
        self.buildBT.pack(side=tkinter.TOP, padx=5, pady=5)
        self.classifyBT = tkinter.Button(self.root, text="Classify", bg="white", width=33, command=self.classifyEvent)
        self.classifyBT.pack(side=tkinter.TOP, padx=5, pady=5)
        self.AccuracyBT = tkinter.Button(self.root, text="Accuracy", bg="white", width=33, command=self.accuracyEvent)
        self.AccuracyBT.pack(side=tkinter.TOP, padx=5, pady=5)
        self.ExitBT = tkinter.Button(self.root, text="Exit", bg="white", width=33, command=self.root.destroy)
        self.ExitBT.pack(side=tkinter.TOP, padx=5, pady=5)
        self.root.geometry('+{0}+{1}'.format(int((self.root.winfo_screenwidth()/2) - (self.root.winfo_reqwidth()/2)), int((self.root.winfo_screenheight()/2) - (self.root.winfo_reqheight()/2))))
        self.root.mainloop()

    def loadFilePathEvent(self):
        """
        function to fulfil event of loading file path from user
        and validate file path has all needed files to continue
        """
        self.folderPath = self.cleaned = self.discretizied = self.builded = self.classified = None
        self.folderPathTF.delete("1.0", tkinter.END)
        self.folderPath = filedialog.askdirectory(initialdir="/", title="Select folder")
        self.folderPathTF.insert(tkinter.END, self.folderPath)
        if self.folderPath == "":
            messagebox.showinfo(title="File path error", message="File path must be entered!" + self.folderPath)
            self.folderPath = None
            return
        self.dataLoader = DataLoader(self.folderPath)
        if not self.dataLoader.validatePath():
            messagebox.showinfo(title="File path error", message="One or more files missing in : " + self.folderPath)
            self.folderPath = None
            return
        if not self.dataLoader.validateNotEmpty():
            messagebox.showinfo(title="File path error", message="One or more files are without content : " + self.folderPath)
            self.folderPath = None
            return

    def cleanEvent(self):
        """
        function to fulfil event of loading and cleaning data from files
        """
        self.cleaned = self.discretizied = self.builded = self.classified = None
        try:
            if not self.folderPath:
                messagebox.showinfo(title="Message", message="Load path before cleaning")
                return
            self.dataLoader.laodStructure()
            self.dataLoader.loadTrain()
            self.dataLoader.loadTest()
            self.fileCreator = CreateFiles(self.folderPath, self.dataLoader.getStructure())
            clean = DataClean(self.fileCreator)
            clean.cleanTrain(self.dataLoader.getTrainData(), self.dataLoader.getStructure())
            clean.cleanTest(self.dataLoader.getTestData(), self.dataLoader.getStructure())
            messagebox.showinfo(title="Message", message=clean.cleaningMessage())
            self.cleaned = True
        except PermissionError:
            messagebox.showinfo(title="Message", message=" Error: Cannot access clean file while open")
            self.cleaned = None
            return

    def discretizationEvent(self):
        """
        function to fulfil event of discretization of data from files
        """
        self.discretizied = self.builded = self.classified = None
        if not self.cleaned:
            messagebox.showinfo(title="Message", message="Clean data before Discretization")
            return
        try:
            self.bins = int(self.discretizationTF.get("1.0", tkinter.END))
            if len(self.dataLoader.getStructure()[len(self.dataLoader.getStructure()) - 1][1]) > self.bins:
                raise ValueError
        except ValueError:
            messagebox.showinfo(title="Message", message="Invalid Discretization bins number")
            return
        try:
            discretizator = DataDiscretization(self.bins, self.fileCreator)
            discretizator.discretizationTrain(self.dataLoader.getTrainData(), self.dataLoader.getStructure())
            discretizator.discretizationTest(self.dataLoader.getTestData(), self.dataLoader.getStructure())
            messagebox.showinfo(title="Message", message=discretizator.discretizationMessage())
            self.discretizied = True
        except PermissionError:
            messagebox.showinfo(title="Message", message=" Error: Cannot access discretization file while open")
            self.cleaned = None
            return

    def buildEvent(self):
        """
        function to fulfil event of building id3 classifier
        """
        self.builded = self.classified = None
        if not self.discretizied:
            messagebox.showinfo(title="Message", message="Do discretization of data before building classifier")
            return
        try:
            self.classifier = Classifier(self.fileCreator, self.bins)
            self.classifier.build(self.dataLoader.getTrainData(), self.dataLoader.getStructure())
            messagebox.showinfo(title="Message", message=self.classifier.buildClassifierMessage())
            self.builded = True
        except PermissionError:
            messagebox.showinfo(title="Message", message=" Error: Cannot access rules file while open")
            self.builded = None
            return

    def classifyEvent(self):
        """
        function to fulfil event of classifying test data by id3 classifier
        """
        self.classified = None
        if not self.builded:
            messagebox.showinfo(title="Message", message="Build classifier before classifying data")
            return
        data = self.dataLoader.getTestData()
        self.newData = createNewData(list(map(lambda x: x[:-1], data)))
        self.classifier.classifyTest(self.newData, self.dataLoader.getStructure())
        messagebox.showinfo(title="Message", message=self.classifier.classifyClassifierMessage())
        self.classified = True

    def accuracyEvent(self):
        """
        function to fulfil event of testing id3 classifier accuracy
        """
        if not self.classified:
            messagebox.showinfo(title="Message", message="Classify data before testing accuracy")
            return
        try:
            accuracy = Accuracy(self.fileCreator, self.bins)
            accuracy.testAccuracy(self.newData, self.dataLoader.getTestData(), self.dataLoader.getStructure())
            messagebox.showinfo(title="Message", message=accuracy.accuracyMessage())
            self.classified = True
        except PermissionError:
            messagebox.showinfo(title="Message", message=" Error: Cannot access accuracy file while open")
            self.classified = None
            return


class DataLoader:
    """
    Class for loading data from file path
    Attributes:
        filePath (String) : the file path of files to load
        structure (list) : nested list of structure file data
        trainData (list) : nested list of train file data
        testData (list) : nested list of test file data
    """
    def __init__(self, filePath):
        """
        The constructor for DataLoader
        Parameters:
            filePath (String) : the file path of files to load
        """
        self.filePath = filePath
        self.structure, self.trainData,  self.testData = [], [], []

    def validatePath(self):
        """
        function to validate path file has test.csv, train.csv, Structure.txt files
        Returns:
            boolean: true if file path has all files else false
        """
        try:
            open(self.filePath + '/test.csv')
            open(self.filePath + '/train.csv')
            open(self.filePath + '/Structure.txt')
        except FileNotFoundError:
            return False
        return True

    def validateNotEmpty(self):
        """
        function to validate test.csv, train.csv, Structure.txt files are not empty
        Returns:
            boolean: true if files not empty else false
        """
        return not (os.stat(self.filePath + '/test.csv').st_size == 0 or os.stat(self.filePath + '/train.csv').st_size == 0 or os.stat(self.filePath + '/Structure.txt').st_size == 0)

    def laodStructure(self):
        """
        function to load Structure.txt file content into a nested list
        each numeric attribute [name,[NUMERIC]]
        each categorical attribute [name,[values]]
        """
        self.structure = []
        with open(self.filePath + '/Structure.txt') as csv_file:
            valuesIndex = 1
            for row in csv_file:
                data = str(row).split()
                data.remove("@ATTRIBUTE")
                columnName = ""
                for i in data[1:-1]:
                    columnName += " " + str(i)
                data[0] = data[0] + columnName
                data = [data[0]] + data[-1:]
                if str(data[valuesIndex]).upper() != "NUMERIC":
                    data[valuesIndex] = data[valuesIndex][1:len(data[1])-1]
                    data[valuesIndex] = data[valuesIndex].split(",")
                self.structure += [data]

    def loadTrain(self):
        """
        function to load train.csv file content into a nested list
        each row is a list [attribute,attribute....]
        """
        self.trainData = []
        with open(self.filePath + '/train.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count != 0:
                    self.trainData += [row]
                line_count += 1

    def loadTest(self):
        """
        function to load test.csv file content into a nested list
        each row is a list [attribute,attribute....]
        """
        self.testData = []
        with open(self.filePath + '/test.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count != 0:
                    self.testData += [row]
                line_count += 1

    def getStructure(self):
        """
        function to get Structure
        Returns:
            list: nested list of structure
        """
        return self.structure

    def getTrainData(self):
        """
        function to get train data
        Returns:
            list: nested list of train data
        """
        return self.trainData

    def getTestData(self):
        """
        function to get test data
        Returns:
            list: nested list of test data
        """
        return self.testData


class DataClean:
    """
    Class for cleaning data from file path
    Attributes:
        fileCreator (CreateFiles) : object for creating new files
        cleanedTrain (boolean) : flag to check if train data has been cleaned
        cleanedTest (boolean) : flag to check if test data has been cleaned
    """
    def __init__(self, fileCreator):
        """
        The constructor for DataClean
        Parameters:
            fileCreator (CreateFiles) : object for creating new files
        """
        self.cleanedTrain, self.cleanedTest, self.fileCreator = False, False, fileCreator

    def cleanTrain(self, trainingData, structure):
        """
        function to clean train data and create new data file if data was cleaned
        """
        self.cleanedTrain = self.removeRowsWithOutClass(trainingData, structure) or self.fillMissingValues(trainingData, structure)
        if self.cleanedTrain:
            self.fileCreator.creatNewCsvFile("/[train]_clean", trainingData)

    def cleanTest(self, testData, structure):
        """
        function to clean test data and create new data file if data was cleaned
        """
        self.cleanedTest = self.fillMissingValues(testData, structure)
        if self.cleanedTest:
            self.fileCreator.creatNewCsvFile("/[test]_clean", testData)

    def removeRowsWithOutClass(self, data, structure):
        """
        function to remove rows with empty class attribute
        Returns:
            boolean: true if rows were removed else false
        """
        removedRowsFlag = False
        for row in data:
            if row[len(structure)-1] == "":
                data.remove(row)
                removedRowsFlag = True
        return removedRowsFlag

    def fillMissingValues(self, data, structure):
        """
        function to fill all types of missing values in rows
        Returns:
            boolean: true if rows were filled else false
        """
        cleanedFlag = False
        for category in structure[0:len(structure)-1]:
            if str(category[1]).upper() == "NUMERIC":
                index = structure.index(category)
                if not cleanedFlag:
                    cleanedFlag = self.fillNumericValues(data, structure, index)
                else:
                    self.fillNumericValues(data, structure, index)
            else:
                index = structure.index(category)
                if not cleanedFlag:
                    cleanedFlag = self.fillCategorialValues(data, index)
                else:
                    self.fillCategorialValues(data, index)
        return cleanedFlag

    def fillNumericValues(self, data, structure, indexCol):
        """
        function to fill numeric missing values in rows
        Returns:
            boolean: true if rows were filled else false
        """
        cleanedFlag = False
        averagesList = self.AverageListByClass(data, structure, indexCol)
        totalAverage = sum(averagesList) / len(averagesList)
        classValues = structure[len(structure) - 1][1]
        for row in data:
            if row[indexCol] == "":
                classValueIndex = list(map(lambda y: int(classValues.index(y)), filter(lambda x: x == row[len(row) - 1], classValues)))
                if len(classValueIndex) == 0:
                    row[indexCol] = totalAverage
                else:
                    row[indexCol] = averagesList[classValueIndex[0]]
                cleanedFlag = True
        return cleanedFlag

    def fillCategorialValues(self, data, indexCol):
        """
        function to fill categorical missing values in rows
        Returns:
            boolean: true if rows were filled else false
        """
        cleanedFlag = False
        columnValues, newColumnValues = list(filter(lambda y: y != "", map(lambda x: x[indexCol], data))), []
        maxValueCount, mostCommonValue = 0, ""
        for i in columnValues:
            if newColumnValues.count(i) == 0:
                newColumnValues.append(i)
                count = columnValues.count(i)
                if maxValueCount < count:
                    maxValueCount = count
                    mostCommonValue = i
        for row in data:
            if row[indexCol] == "":
                row[indexCol] = mostCommonValue
                cleanedFlag = True
        return cleanedFlag

    def AverageListByClass(self, data, structure, indexCol):
        """
        function to get averages of column by class attribute
        Returns:
            list: averages by class values order in structure
        """
        averages = []
        for value in structure[len(structure)-1][1]:
            newData = list(filter(lambda x: x[len(x)-1] == value, data))
            columnData = list(map(lambda z: float(z), filter(lambda y: y != "", map(lambda x: x[indexCol], newData))))
            average = reduce(lambda x, y: x + y, columnData) / len(columnData)
            averages += [average]
        return averages

    def cleaningMessage(self):
        """
        function to get cleaning message content
        Returns:
            String : "Loading DataSet Done and Cleaning DataSet Done" if data was cleaned else "Loading DataSet Done"
        """
        return "Loading DataSet Done and Cleaning DataSet Done" if (self.cleanedTrain or self.cleanedTest) else "Loading DataSet Done"


class DataDiscretization:
    """
    Class for cleaning data from file path
    Attributes:
        fileCreator (CreateFiles) : object for creating new files
        discretizatedTest (boolean) : flag to check if occurred discretization of train data
        discretizatedTrain (boolean) : flag to check if occurred discretization of test data
        bins (int) : number of discretization bins
        binsList (list): list of bins ranges for changing values to categories
    """
    def __init__(self, bins, fileCreator):
        """
        The constructor for DataDiscretization
        Parameters:
            fileCreator (CreateFiles) : object for creating new files
            bins (int) : number of discretization bins
        """
        self.bins, self.fileCreator, self.binsList = bins, fileCreator, []
        self.discretizatedTest, self.discretizatedTrain = False, False

    def discretizationTrain(self, trainData, structure):
        """
        function to do discretization of train data and create new data file
        Parameters:
            trainData (list): the data of train file
            structure (list): the structure of data
        """
        def sortBySplitValue(value):
            """
            function to get element for sorting list
            Parameters:
                value (list): atrribute for sorting place
            Returns:
                float: value for sorting
            """
            return float(value[0])
        statistic = Statistics()
        for i in structure:
            if i[1] == "NUMERIC":
                self.discretizatedTrain = True
                gainList = statistic.bestGainSplits(trainData, structure, structure.index(i))
                gainList = gainList[0:self.bins-1]
                gainList.sort(reverse=True, key=sortBySplitValue)
                binsList = ["> " + str(gainList[0][0])]
                for k in trainData:
                    if float(k[structure.index(i)]) > gainList[0][0]:
                        k[structure.index(i)] = binsList[0]
                for j in range(1, len(gainList)):
                    binsList += [str(gainList[j][0])+" to "+str(gainList[j-1][0])]
                    for k in trainData:
                        try:
                            value = float(k[structure.index(i)])
                            if gainList[j][0] < value <= gainList[j-1][0]:
                                k[structure.index(i)] = binsList[j]
                        except ValueError:
                            pass
                binsList += ["<= " + str(gainList[len(gainList)-1][0])]
                for k in trainData:
                    try:
                        value = float(k[structure.index(i)])
                        if value <= gainList[len(gainList)-1][0]:
                            k[structure.index(i)] = binsList[len(gainList)]
                    except ValueError:
                        pass
                self.binsList += [binsList]
        if self.discretizatedTrain:
            self.fileCreator.creatNewCsvFile("/[train]__discretization_[#" + str(self.bins) + "]", trainData)

    def discretizationTest(self, testData, structure):
        """
        function to do discretization of test data and create new data file
        Parameters:
            trainData (list): the data of train file
            structure (list): the structure of data
        """
        index = 0
        for i in structure:
            if i[1] == "NUMERIC":
                self.discretizatedTest = True
                attributes = self.binsList[index]
                for k in testData:
                    if float(k[structure.index(i)]) > float(str(attributes[0]).replace(">", "")):
                        k[structure.index(i)] = attributes[0]
                for j in range(1, len(attributes)-1):
                    Range = str(attributes[j]).split()
                    Range.pop(1)
                    for k in testData:
                        try:
                            value = float(k[structure.index(i)])
                            if float(Range[0]) < value <= float(Range[1]):
                                k[structure.index(i)] = attributes[j]
                        except ValueError:
                            pass

                for k in testData:
                    try:
                        value = float(k[structure.index(i)])
                        if value <= float(str(attributes[len(attributes) - 1]).replace("<=", "")):
                            k[structure.index(i)] = attributes[len(attributes)-1]
                    except ValueError:
                        pass
                structure[structure.index(i)][1] = self.binsList[index]
                index += 1
        if self.discretizatedTest:
            self.fileCreator.creatNewCsvFile("/[test]__discretization_[#" + str(self.bins) + "]", testData)

    def discretizationMessage(self):
        """
        function to get discretization message content
        Returns:
            String : "Discretization DataSet Using [#bins] Bins Done" if data was cleaned else "No discretization needed"
        """
        if self.discretizatedTest or self.discretizatedTrain:
            return "Discretization DataSet Using [#" + str(self.bins) + "] Bins Done"
        return "No discretization needed"


class Classifier:
    """
    Class for creating id3 classifier from train data and Classifying test data
    Attributes:
        fileCreator (CreateFiles) : object for creating new files
        bins (int) : number of discretization bins
    """
    def __init__(self, fileCreator, bins):
        """
        The constructor for Classifier
        Parameters:
            fileCreator (CreateFiles) : object for creating new files
            bins (int) : number of discretization bins
        """
        self.fileCreator = fileCreator
        self.bins = bins

    def build(self, trainData, structure):
        """
        finction to create id3 classifier and create rules file
        Parameters:
            trainData (list): the data of train file
            structure (list): the structure of data
        """
        self.tree = Node("root", "rule", self.ID3(trainData, structure, createNewStructure(structure), self.mostCommonClassAttribute(trainData)))
        print("finished tree!!!!!!!!!!")
        self.PostPruningTree(trainData, self.tree, structure)
        print("finished prone!!!!!!!!!!")
        self.buildRules()
        print("finished build rules!!!!!!!")
        self.fileCreator.creatNewTxtRuleFile("/id3_rules_discretization_[#" + str(self.bins) + "]", self.rules)

    def ID3(self, data, structure, changableStructure, mostCommonClassAttribute):
        """
        function to fulfil id3 algorithm
        Parameters:
            data (list): the data for the algorithm
            structure (list): the structure of data
            changableStructure(list): the structure of data that changes through the algorithm
            mostCommonClassAttribute(String): the most common class attribute in data
        Returns:
            Node: tree of id3
        """
        if len(data) == 0:
            return [Node("class", mostCommonClassAttribute)]
        if len(changableStructure[0:len(changableStructure)-1]) == 0 or len(data) == 1:
            return [Node("class", self.mostCommonClassAttribute(data))]
        for i in range(0, len(data)-1):
            if data[i][len(data[i])-1] != data[i+1][len(data[i+1])-1]:
                break
            elif i == len(data)-1:
                return [Node("class", data[0][len(data[0]-1)])]

        root = self.bestInfoGain(data, structure, changableStructure)
        indexOfCol = int(list(map(lambda y: structure.index(y), filter(lambda x: x[0] == root, structure)))[0])
        values = list(map(lambda y: y[1], filter(lambda x: x[0] == root, structure)))[0]
        changableStructure.remove(list(filter(lambda x: x[0] == root, changableStructure))[0])
        mostCommonClassAttribute = self.mostCommonClassAttribute(data)
        subsList = []
        for i in values:
            newData = list(filter(lambda x: x[indexOfCol] == i, data))
            son = Node(root, i)
            son.addSubNodes(self.ID3(newData, structure, createNewStructure(changableStructure), mostCommonClassAttribute))
            subsList += [son]
        return subsList

    def bestInfoGain(self, data, structure, changableStructure):
        """
        function to get the attribute with best info-gain of column
        Parameters:
            data (list): the data for the algorithm
            structure (list): the structure of data
            changableStructure(list): the structure of data that changes through the algorithm
        Returns:
            String: the best info-gain attribute
        """
        statistics = Statistics()
        maxInfoGain = 0
        gainCol = ""
        for i in changableStructure[0:len(changableStructure)-1]:
            infoGain = statistics.infoGainOfColoumn(data, structure, structure.index(i))
            if infoGain >= maxInfoGain:
                maxInfoGain = infoGain
                gainCol = i[0]
        return gainCol

    def mostCommonClassAttribute(self, data):
        """
        function to get most most common class attribute of data
        Parameters:
            data (list): the data for the algorithm
        Returns:
            String: the most common class attribute of data
        """
        values, mostCommon, maxCount = [], "", 0
        for i in data:
            if values.count(i[len(i)-1]) == 0:
                values.append(i[len(i)-1])
        for i in values:
            count = list(map(lambda x: x[len(data[0])-1], data)).count(i)
            if maxCount < count:
                mostCommon = i
                maxCount = count
        return mostCommon

    def buildRules(self):
        """
        function to build list of rules
        """
        path, allPaths = [], []

        def rules(tree, path, allPaths):
            """
            recursive function to convert id3 tree to list of rules
            Parameters:
                tree (Node): id3 tree to convert to list
                path (list): list that represents a branch in tree
                allPaths (list): all branches in tree
            """
            if tree.subNodes is None:
                path.append(tree.name + " == " + tree.value)
                allPaths += [path]
                return
            else:
                path.append(tree.name + " == " + tree.value)
                for i in tree.subNodes:
                    rules(i, path.copy(), allPaths)
            return
        rules(self.tree, path, allPaths)
        for i in allPaths:
            allPaths[allPaths.index(i)] = i[1:len(i)]
        self.rules = allPaths

    def PostPruningTree(self, data, tree, structure):
        """
        function to post pruning id3 tree
        Parameters:
            data(list): the data of train
            tree (Node): the tree to post prune
            structure(list): the structure of data
        """
        def PostPruningSubTree(data, tree, structure):
            """
            recursive fuction to post pruning id3 sub tree
            Parameters:
                data(list): the data of branch rule
                tree (Node): the sub tree to post prune
                structure(list): the structure of data
            """
            sumNumerator, sumDenominator, newData = 0, 0, []
            if tree.subNodes is None:
                return True
            print(tree.subNodes)
            for j in tree.subNodes:
                indexCol = int(list(map(lambda y: structure.index(y), filter(lambda x: x[0] == tree.name, structure)))[0])
                newData = list(filter(lambda x: x[indexCol] == tree.value, data))
                flag = PostPruningSubTree(newData, j, structure)
                if flag:
                    return
                sumNumerator += self.calcNumerator(newData)
                sumDenominator += len(newData)
            qV = self.calcNumerator(data)/len(data)
            qT = sumNumerator/sumDenominator
            if qV <= qT:
                tree.subNodes = [Node("class", self.mostCommonClassAttribute(newData))]
        for i in tree.subNodes:
            PostPruningSubTree(data, i, structure)
        return False

    def calcNumerator(self, data):
        """
        function to calc numerator of error of node if pruned
        Returns:
            float: the numerator of error of node if pruned
        """
        N = len(data)
        Nc = len(list(filter(lambda y: y[len(y)-1] == self.mostCommonClassAttribute(data), data)))
        return N - Nc + 0.5

    def classifyTest(self, testData, structure):
        """
        function to classify test data
        Parameters:
            testData(list): the data of test to classify
            structure(list): the structure of data
        """
        for i in testData:
            testData[testData.index(i)] += [self.testAttribute(i, structure)]
        self.fileCreator.creatNewCsvFile('/test_class_discretization_[#' + str(self.bins) + 'bins]', testData)

    def testAttribute(self, attr, structure):
        """
        function to classify a row
        Parameters:
            attr(list): the row to classify
            structure(list): the structure of data
        """
        for i in self.rules:
            flag = True
            for j in i[0:len(i)-1]:
                rule = j.split("==")
                for k in range(0, len(rule)):
                    rule[k] = rule[k].strip()
                if attr[int(list(map(lambda y: structure.index(y), filter(lambda x: x[0] == rule[0], structure)))[0])] != rule[1]:
                    flag = False
            if flag:
                result = i[len(i)-1].split("==")
                return result[1].strip()

    def buildClassifierMessage(self):
        """
        function to get build classifier message
        Returns:
            string: "Building classifier using train-set is done"
        """
        return "Building classifier using train-set is done"

    def classifyClassifierMessage(self):
        """
        function to get classification message
        Returns:
            string: "Classifying the test-set is done!"
        """
        return "Classifying the test-set is done!"


class Accuracy:
    """
    Class for testing algorithm accuracy
    Attributes:
        fileCreator (CreateFiles) : object for creating new files
        bins (int) : number of discretization bins
        accuracyPercent(float): accuracy percent
    """
    def __init__(self, fileCreator, bins):
        """
        The constructor for Accuracy
        Parameters:
            fileCreator (CreateFiles) : object for creating new files
            bins (int) : number of discretization bins
        """
        self.accuracyPercent = 0.0
        self.fileCreator = fileCreator
        self.bins = bins

    def testAccuracy(self, newData, oldData, structure):
        """
        function to test accuracy of are classification and create accuracy file
        Parameters:
            newData(list): the test data we classified
            oldData(list): the test data with class attributes
            structure(list): the structure of data
        """
        errorsCount, classIndex, numberOfattrs = 0, len(structure)-1, len(newData)
        indexsOfRowErrors = []
        for rowIndex in range(0, len(newData)):
            if newData[rowIndex] != oldData[rowIndex]:
                errorsCount += 1
                indexsOfRowErrors += [(rowIndex+1)]
        self.accuracyPercent = ((numberOfattrs-errorsCount)/numberOfattrs) * 100
        self.fileCreator.createNewTxtAccuracyFile('/test_accuracy_discretization_[#' + str(self.bins) + '].txt', [self.accuracyMessage()]+indexsOfRowErrors)


    def accuracyMessage(self):
        """
        function to get accuracy message
        Returns:
            string: "accuracy is:  accuracyPercent %"
        """
        return "accuracy is: " + str(self.accuracyPercent) + "%"


class CreateFiles:
    """
    class to create files
    Attributes:
        filePath (String): path to create files
        structure (list): structure of data
    """
    def __init__(self, filePath, structure):
        """
        The constructor for CreateFiles
        Parameters:
            filePath (String): path to create files
            structure (list): structure of data
        """
        self.filePath = filePath
        self.structure = structure

    def creatNewCsvFile(self, name, data):
        """
        function to create a new csv file of data by name
        Parameters:
            name (String): file name
            data (list): the data in the file
        """
        with open(self.filePath + name + '.csv', mode='w', newline='') as csvFile:
            fileWriter = csv.writer(csvFile, delimiter=',')
            fileWriter.writerow(list(map(lambda x: x[0], self.structure)))
            fileWriter.writerows(data)

    def creatNewTxtRuleFile(self, name, data):
        """
        function to create a new txt file of classification rules by name
        Parameters:
            name (String): file name
            data (list): the rules of classification
        """
        with open(self.filePath + name + '.txt', mode='w', newline='') as txtFile:
            for i in data:
                string = "If "
                for j in i[0:len(i)-1]:
                    string += j
                    if i.index(j) != len(i)-2 :
                       string += " and "
                string += " then " + i[len(i)-1]
                txtFile.write(string + '\n')

    def createNewTxtAccuracyFile(self, name, data):
        """
        function to create a new txt file of classification accuracy by name
        Parameters:
            name (String): file name
            data (list): indexes of incorrect classification rows
        """
        with open(self.filePath + name + '.txt', mode='w', newline='') as txtFile:
            for i in range(0, len(data)):
                if i == 0:
                    txtFile.write(str(data[i]) + '\n')
                    txtFile.write("Indexes of incorrect classification: ")
                else:
                    txtFile.write(str(data[i]))
                    if i != len(data)-1:
                        txtFile.write(",")


class Statistics:
    """
    class of statistics calculations
    """
    def __init__(self):
        pass

    def entropyClass(self, data, structure):
        """
        function to calculate the entropy of a class in data
        Parameters:
            data (list): the data in the file
            structure (list): the structure of data
        Returns:
            float: the result of the class entropy
        """
        sizeOfData = len(data)
        result = 0
        for i in structure[len(structure) - 1][1]:
            p = len(list(filter(lambda x: x[len(structure) - 1] == i, data))) / sizeOfData
            p = 1 if p == 0 else p
            result += (-1) * (p * math.log(p, 2))
        return result

    def entropyOfSplit(self, data, structure, indexOfCol, split):
        """
        function to calculate the entropies below and above of a split
        Parameters:
            data (list): the data in the file
            structure (list): the structure of data
            indexOfCol (int): index of column for calculating entropies
            split (float): the split to calculate entropy below and above
        Returns:
            list: list with entropies below and above the split [below,above]
        """
        sizeOfBeforeSplit = len(list(filter(lambda x: float(x[indexOfCol]) <= split, data)))
        sizeOfAfterSplit = len(list(filter(lambda x: float(x[indexOfCol]) > split, data)))

        entropyBelow = 0
        entropyAbove = 0
        for i in structure[len(structure) - 1][1]:
            PBelow = 0 if sizeOfBeforeSplit == 0 else len(list(filter(lambda x: x[len(structure) - 1] == i and float(x[indexOfCol]) <= split, data))) / sizeOfBeforeSplit
            PAbove = 0 if sizeOfAfterSplit == 0 else len(list(filter(lambda x: x[len(structure) - 1] == i and float(x[indexOfCol]) > split, data))) / sizeOfAfterSplit
            PBelow = 1 if PBelow == 0 else PBelow
            PAbove = 1 if PAbove == 0 else PAbove
            entropyBelow += (-1) * (PBelow * math.log2(PBelow))
            entropyAbove += (-1) * (PAbove * math.log2(PAbove))

        return [entropyBelow, entropyAbove]

    def infoGainOfSplit(self, data, structure, indexOfCol, split):
        """
        function to calculate info-gain of a split
        Parameters:
            data (list): the data in the file
            structure (list): the structure of data
            indexOfCol (int): index of column for calculating info-gain
            split (float): the split to calculate info-gain
        Returns:
            float: info-gain of a split
        """
        entropies = self.entropyOfSplit(data, structure, indexOfCol, split)
        sizeOfData = len(data)
        sizeOfBeforeSplit = len(list(filter(lambda x: float(x[indexOfCol]) <= split, data)))
        sizeOfAfterSplit = len(list(filter(lambda x: float(x[indexOfCol]) > split, data)))
        info = entropies[0] * (sizeOfBeforeSplit / sizeOfData) + entropies[1] * (sizeOfAfterSplit / sizeOfData)
        gain = self.entropyClass(data, structure) - info
        return gain

    def infoGainOfColoumn(self, data, structure, indexOfCol):
        """
        function to calculate info-gain of a column
        Parameters:
            data (list): the data in the file
            structure (list): the structure of data
            indexOfCol (int): index of column for calculating info-gain
        Returns:
            float: info-gain of a column
        """
        classEntropy = self.entropyClass(data, structure)
        info = 0
        X, Y, Z, p = 0, 0, 0, 0
        for i in structure[indexOfCol][1]:
            entropy = 0
            Y = len(list(filter(lambda x: x[indexOfCol] == i, data)))
            for j in structure[len(structure) - 1][1]:
                X = len(list(filter(lambda x: x[len(structure)-1] == j and x[indexOfCol] == i, data)))
                p = 1 if Y == 0 else X/Y
                p = 1 if p == 0 else p
                entropy += (-1) * p * math.log2(p)
            Z = len(data)
            info += (Y/Z) * entropy
        ans = classEntropy - info
        ans = 0 if ans < 0 else ans
        return ans

    def listGainsOfSplit(self, data, structure, indexOfCol):
        """
        function to create a list of all splits and their gains in data
        Parameters:
            data (list): the data in the file
            structure (list): the structure of data
            indexOfCol (int): index of column for calculating list of info-gain by split
        Returns:
            list: list of splits and info-gains [[split,info-gain],....]
        """
        def sortColoumn(val):
            """
            function to get element for sorting list
            Parameters:
                val (list): atrribute for sorting place
            Returns:
                float: value for sorting
            """
            return float(val[indexOfCol])
        data = createNewData(data)
        data.sort(key=sortColoumn)
        gainList = []
        splitInGainList = 0
        for i in range(0, len(data)-1):
            split = (float(data[i][indexOfCol]) + float(data[i+1][indexOfCol])) / 2
            if i != 0:
                splitInGainList = len(list(filter(lambda x: x[0] == split, gainList)))
            if splitInGainList == 0:
                gainList.append([split, self.infoGainOfSplit(data, structure, indexOfCol, split)])
        indexOfCol = 1
        gainList.sort(reverse=True, key=sortColoumn)
        return gainList

    def bestGainSplits(self, data, structure, indexCol):
        """
        recursive function to get splits by highest gain
        Parameters:
            data (list): the data to check the best split
            structure (list): the structure of data
            indexOfCol (int): index of column for calculating list of best info-gain by split
        Returns:
            list: list of splits and info-gains [[split,info-gain],....]
        """
        gain = self.listGainsOfSplit(data, structure, indexCol)
        if len(gain) == 0:
            return []
        if gain[0][1] == 0:
            return [gain[0]]
        if len(list(filter(lambda x: float(x[indexCol]) == gain[0][0], data))) == len(data):
            return [gain[0]]
        newDataAbove = list(filter(lambda x: float(x[indexCol]) > gain[0][0], data))
        newDataBelow = list(filter(lambda x: float(x[indexCol]) <= gain[0][0], data))
        value1 = self.bestGainSplits(newDataAbove, structure, indexCol)
        value2 = self.bestGainSplits(newDataBelow, structure, indexCol)
        if len(value1) != 0 and len(value2) != 0:
            if value1[0][1] >= value2[0][1]:
                return [gain[0]] + value1 + value2
            else:
                return [gain[0]] + value2 + value1
        elif len(value1) != 0:
            return [gain[0]] + value1
        elif len(value2) != 0:
            return [gain[0]] + value2
        return [gain[0]]


class Node:
    """
    class of Node in tree
    Attributes:
        name (String): the column name
        value (String): value in column node
        subNodes (list): list of sons of node
    """
    def __init__(self, name, value, subNodes=None):
        """
        The constructor for class Node
        Parameters:
            name (String): the column name
            value (String): value in column node
            subNodes (list): list of sons of node
        """
        self.name = name
        self.value = value
        self.subNodes = subNodes

    def addSubNodes(self, nodeList):
        """
        function to add sons to a node
        Parameters:
            nodeList (list): son to add to current node
        """
        if self.subNodes is None:
            self.subNodes = []
        self.subNodes += nodeList

    def getSubNodes(self):
        """
        function to get the sons of a node
        Returns:
            (list): sons of current node
        """
        return self.subNodes


def createNewStructure(structure):
    """
    function to create new structure list
    Parameters:
        structure (list): structure to copy
    Returns:
        (list): new structure list
    """
    newStructure = []
    for i in structure:
        newStructure += [i.copy()]
    return newStructure


def createNewData(data):
    """
    function to create new data list
    Parameters:
        data (list): data to copy
    Returns:
        (list): new data list
    """
    newData= []
    for i in data:
        newData += [i.copy()]
    return newData


def getColumnValues(data,index):
    """
    function to get values in a column
    Parameters:
        index (int): index of a column
    Returns:
        (list): values of a column
    """
    values = list(map(lambda x: x[index], data))
    newValues = []
    for i in values:
        if newValues.count(i) == 0:
            newValues.append(i)
    return newValues


if __name__ == "__main__":
    Gui()
