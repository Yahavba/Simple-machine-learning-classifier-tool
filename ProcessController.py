from DataClassifier import Classifier
from DataCleaner import Cleaner
from DataDiscretization import Discretization
from DataLoader import Loader
from MiningCalculations import MiningCalculator
from FileCreator import CreateFile


class BuildClassifierProcess:
    """ class for building a clasiifier and doing all needed processes (cleaning, discretization and more)"""
    def __init__(self):
        pass

    def setClassifierType(self, classifierType):
        """
        method to set classifier type for process
        Attributes:
            classifierType(String) : name of classifier
        Returns:
            BuildClassifierProcess: the object we set
        """
        self.classifierType = classifierType
        return self

    def setClassifierSplitType(self, classifierSplitType):
        """
        method to set classifier split type for process
        Attributes:
            classifierSplitType(String) : name of classifier split type
        Returns:
            BuildClassifierProcess: the object we set
        """
        self.classifierSplitType = classifierSplitType
        return self

    def setDiscretizationType(self, discretizationType):
        """
        method to set discretization type for process
        Attributes:
            discretizationType(String) : name of discretization type
        Returns:
            BuildClassifierProcess: the object we set
        """
        self.discretizationType = discretizationType
        return self

    def setDiscretizationBins(self, discretizationBins):
        """
        method to set number of discretization bins for process
        Attributes:
            discretizationBins(int) : number of discretization bins
        Returns:
            BuildClassifierProcess: the object we set
        """
        self.discretizationBins = discretizationBins
        return self

    def setFolderPath(self, folderPath):
        """
        method to set csv file path with data for process
        Attributes:
            folderPath(String) : csv file path with data
        Returns:
            BuildClassifierProcess: the object we set
        """
        self.folderPath = folderPath
        return self

    def setSavingFolderPath(self, savingFolderPath):
        """
        method to set folder path for saving files in process
        Attributes:
            savingFolderPath(String) : folder path for saving files
        Returns:
            BuildClassifierProcess: the object we set
        """
        self.savingFolderPath = savingFolderPath
        return self

    def startProcess(self, labelWidget):
        """
        method to start process after all setters have been activated
        Attributes:
            labelWidget(tkinter.Label) : a message box for showing process to user
        """
        fileCreator, dataCleaner, dataDiscretization, Calculator, dataClassifier, dataLoader = CreateFile(), Cleaner(), Discretization(),\
                                                                                               MiningCalculator(), Classifier(), Loader()
        splitFunction = Calculator.getSplitFunc(self.classifierSplitType)

        try:
            labelWidget.configure(text=labelWidget.cget("text") + "Building process starting\n")

            dataLoader.loadData(self.folderPath)
            labelWidget.configure(text=labelWidget.cget("text") + "Data loading Finished\n")

            dataCleaner.cleanTrainingSet(dataLoader.trainingSet, dataLoader.structure)
            dataCleaner.cleanTestSet(dataLoader.testSet, dataLoader.structure)
            fileCreator.createCsvFile(dataLoader.structure, dataLoader.trainingSet, "Clean Training set", self.savingFolderPath)
            fileCreator.createCsvFile(dataLoader.structure, dataLoader.testSet, "Clean Test set", self.savingFolderPath)
            labelWidget.configure(text=labelWidget.cget("text") + "Data cleaning Finished\n")

            dataDiscretization.discretizationData(dataLoader.trainingSet, dataLoader.testSet, dataLoader.structure, self.discretizationBins,
                                                  self.discretizationType)
            fileCreator.createCsvFile(dataLoader.structure, dataLoader.trainingSet, "Discretization Training set", self.savingFolderPath)
            fileCreator.createCsvFile(dataLoader.structure, dataLoader.testSet, "Discretization Test set", self.savingFolderPath)
            labelWidget.configure(text=labelWidget.cget("text") + "Data Discretization Finished\n")

            classifier = dataClassifier.buildClassifier(dataLoader.trainingSet, dataLoader.structure, self.classifierType, splitFunction)
            classifiedTestData = dataClassifier.classifyTest(dataLoader.testSet, dataLoader.structure, classifier)
            fileCreator.createCsvFile(dataLoader.structure,  classifiedTestData, "Classified Test set", self.savingFolderPath)
            accuracy = dataClassifier.checkAccuracyOfClassifier(classifiedTestData, dataLoader.testSet)
            classifier += ["accuracy: " + str(accuracy)]
            fileCreator.createTxtFile(classifier, "Rules", self.savingFolderPath)
            labelWidget.configure(text=labelWidget.cget("text") + "Building classifier Finished\n")

            return labelWidget.configure(text=labelWidget.cget("text") + "Classifier build successfully with accuracy: " + str(round(accuracy, 3)) +
                                              "\n")

        except EnvironmentError:
            return labelWidget.configure(text=labelWidget.cget("text") +
                                              "Problem with file\\ file path. please check file is not empty and file path is correct!")
        except:
            return labelWidget.configure(text=labelWidget.cget("text") +
                                              "An Error occurred please check file and inputs and start again!")


