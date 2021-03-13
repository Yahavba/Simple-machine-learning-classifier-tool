from MiningCalculations import MiningCalculator
import copy


class Classifier:
    def __init__(self):
        """"
        Ctor for Classifier
        """
        self.calculator = MiningCalculator()

    def buildClassifier(self, data, structure, classifierType, splitFunc=None):
        """
        method to build classifier
        Attributes:
            data(list) : list of lines in files each element is a list
            structure(dict): the structure of data set returns {} if data set is empty, each element is
                            columnName : {'index': index , 'values': [values]} or
                            columnName : {'index': index , 'values': ["Numeric"]
            classifierType(String): the type of classifier to build
            splitFunc (function): a split method
        Returns:
            list: a list with rules each element is a rule
        """
        if classifierType.upper() == "ID3":
            rules = self.buildId3Classifier(data, structure, self.calculator.calcNumberOfMajorityClassRows(data, structure), splitFunc)
            return rules
        elif classifierType.upper() == "NAIVE BAYES":
            rules = self.buildNaiveBayesClassifier(structure, data)
            return rules

    # ID3 Classifier

    def buildId3Classifier(self, data, structure, mostCommonClassAttribute, splitFunc):
        """
        method to build Id3 classifier (rules) by a split method
        Attributes:
            data(list) : list of lines in files each element is a list
            structure(dict): the structure of data set returns {} if data set is empty, each element is
                            columnName : {'index': index , 'values': [values]} or
                            columnName : {'index': index , 'values': ["Numeric"]
            mostCommonClassAttribute(String): the most most common class attribute in data
            splitFunc (function): a split method
        Returns:
            list: a list of rules
        """
        tree = self.buildId3Tree(data, structure, mostCommonClassAttribute, splitFunc)
        self.postPruneTree(data, structure, tree)
        rulesList = self.ExtractRulesFromId3Tree(tree)
        return rulesList

    def buildId3Tree(self, data, structure, mostCommonClassAttribute, splitFunc):
        """
        method to build DecisionTree by ID3 algorithm
        Attributes:
            data(list) : list of lines in files each element is a list
            structure(dict): the structure of data set returns {} if data set is empty, each element is
                            columnName : {'index': index , 'values': [values]} or
                            columnName : {'index': index , 'values': ["Numeric"]
            mostCommonClassAttribute(String): the most most common class attribute in data
            splitFunc (function): a split method
        Returns:
            list: a list of first sub trees of id3 algorithm example [tree1, tree2, tree3]
        """
        if len(data) == 0:
            return [DecisionTree("class", mostCommonClassAttribute)]
        if len(structure) - 1 == 0 or self.calculator.allRowsWithSameClass(data, structure):
            return [DecisionTree("class", self.calculator.mostCommonClassAttribute(data, structure))]

        root = splitFunc(data, structure)
        values, subsList = structure[root]['values'], []
        mostCommonClassAttribute = self.calculator.mostCommonClassAttribute(data, structure)
        for val in values:
            newData = list(filter(lambda x: x[structure[root]['index']] == val, data))
            Node = DecisionTree(root, val, len(newData), self.calculator.calcNumberOfMajorityClassRows(newData, structure))
            Node.addSubDecisionTree(self.buildId3Tree(newData, self.createNewStructureWithoutItem(structure, root),
                                                      mostCommonClassAttribute, splitFunc))
            subsList += [Node]
        return subsList

    def postPruneTree(self, data, structure, treeList):
        """
        method to post pruning DecisionTree tree
        Attributes:
            data(list) : list of lines in files each element is a list
            structure(dict): the structure of data set returns {} if data set is empty, each element is
                            columnName : {'index': index , 'values': [values]} or
                            columnName : {'index': index , 'values': ["Numeric"]
            treeList (list): a list of first sub trees of id3 algorithm example [tree1, tree2, tree3]
        """
        def prune(data, structure, tree):
            """
            method to prune a DecisionTree
            Attributes:
                data(list) : list of lines in files each element is a list
                structure(dict): the structure of data set returns {} if data set is empty, each element is
                            columnName : {'index': index , 'values': [values]} or
                            columnName : {'index': index , 'values': ["Numeric"]
                tree (DecisionTree): DecisionTree to post prune
            """
            QtNumerator, QtDenominator, newData = 0, 0, []
            if tree.SubDecisionTree is None:
                return True
            for j in tree.SubDecisionTree:
                indexCol = structure[tree.name]['index']
                newData = list(filter(lambda x: x[indexCol] == tree.value, data))
                flag = prune(newData, structure, j)
                if flag:
                    return
                QtNumerator += (j.N - j.Nc + 0.5)
                QtDenominator += j.N
            qV = (tree.N - tree.Nc + 0.5) / tree.N
            qT = QtNumerator/QtDenominator
            if qV <= qT:
                tree.SubDecisionTree = [DecisionTree("class", self.calculator.mostCommonClassAttribute(newData, structure))]
        for i in treeList:
            prune(data, structure, i)

    def ExtractRulesFromId3Tree(self, treeList):
        """
        method to build list of rules from a DecisionTree
        Attributes:
            treeList(list): a list of first sub trees of id3 algorithm example [tree1, tree2, tree3]
        Returns:
            list: all rules in tree
        """
        rule, allRules = "", []

        def rules(tree, rule, allRules):
            """
            recursive method to convert DecisionTree tree to list of rules
            Attributes:
                tree (DecisionTree): DecisionTree tree to convert to list
                rule (String): string that represents a branch in tree
                allRules (list): all branches in tree
            """
            if tree.SubDecisionTree is None:
                rule = rule[:-3]
                rule += " => " + tree.name + " == " + tree.value
                allRules += [rule]
                return
            else:
                rule += tree.name + " == " + tree.value + " , "
                for subTree in tree.SubDecisionTree:
                    rules(subTree, rule, allRules)
            return
        for i in treeList:
            rules(i, rule, allRules)
        return allRules

    # naive Bayes classifier

    def buildNaiveBayesClassifier(self, structure, data):
        """
        method to build rules by naive bayes classifier
        Attributes:
                data(list) : list of lines in files each element is a list
                structure(dict): the structure of data set returns {} if data set is empty, each element is
                            columnName : {'index': index , 'values': [values]} or
                            columnName : {'index': index , 'values': ["Numeric"]
        Returns:
            list:  list of rules each element is a s string rule
        """
        rules = []
        ProbabilityDict = self.createProbabilityDict(structure, data)
        combinations = self.createColumnValuesCombination(structure)
        classValues = structure['class']['values']
        for combination in combinations:
            flag, rule = False, ""
            for i in combination:
                if flag:
                    rule += ", " + i.replace('=>', ' == ')
                else:
                    rule += i.replace('=>', ' == ')
                    flag = True
            rule += " => class" + " == " + self.classOfCombination(data, combination, ProbabilityDict, classValues)
            rules += [rule]
        return rules

    def classOfCombination(self, data, combination, ProbabilityDict, classValues):
        """
        method to find the class value of a combination
        Attributes:
            combination(list): combination of column values
            ProbabilityDict(Dict) : dictionary with Probability of value given class example {class value: {column value: probability}...}
            classValues(list): values of class
        Returns:
            String:  class value of a combination
        """
        maxProbability, classOfCombination = 0, None
        for classValue in classValues:
            Probability = self.calcProbabilityOfCombinationGivenClass(combination, ProbabilityDict, classValue)
            Probability *= self.calculator.calcProbabilityOfClassValueWithLaplaceCorrection(data, classValue, len(classValues))
            if Probability >= maxProbability:
                maxProbability = Probability
                classOfCombination = classValue
        return classOfCombination

    def calcProbabilityOfCombinationGivenClass(self, combination, ProbabilityDict, classValue):
        """
        method to calc p(x| ci) x is a combination and ci is a class value
        Attributes:
            combination(list): combination of column values
            ProbabilityDict(Dict) : dictionary with Probability of value given class example {class value: {column value: probability}...}
            classValue(String): value of class
        Returns:
            float:  p(x| ci) x is a combination and ci is a class value
        """
        probability = 1
        for element in combination:
            for name, probabilityValue in ProbabilityDict[classValue].items():
                if element == name:
                    probability *= probabilityValue
        return round(probability, 3)

    def createProbabilityDict(self, structure, data):
        """
        method to create a dictionary with Probability of value given class
        Attributes:
            data(list) : list of lines in files each element is a list
            structure(dict): the structure of data set returns {} if data set is empty, each element is
                        columnName : {'index': index , 'values': [values]} or
                        columnName : {'index': index , 'values': ["Numeric"]
        Returns:
            Dict: dictionary with Probability of value given class example {class value: {column value: probability}...}
        """
        probabilityDict, classValues = {}, structure['class']['values']
        for column, values in structure.items():
            if column != 'class':
                for value in values['values']:
                    for classValue in classValues:
                        if classValue not in probabilityDict:
                            probabilityDict[classValue] =\
                                {column + '=>' + value:
                                     self.calculator.calcProbabilityOfValGivenClassWithLaplaceCorrection(data,
                                                                                                         values['index'], value, classValue,
                                                                                                         len(values['values']))}
                        else:
                            probabilityDict[classValue][column + '=>' + value] = \
                                self.calculator.calcProbabilityOfValGivenClassWithLaplaceCorrection(data, values['index'], value, classValue,
                                                                                                    len(values['values']))
        return probabilityDict

    def createColumnValuesCombination(self, structure):
        """
        method to create all combination of column values
        Attributes:
            structure(dict): the structure of data set returns {} if data set is empty, each element is
                        columnName : {'index': index , 'values': [values]} or
                        columnName : {'index': index , 'values': ["Numeric"]
        Returns:
            list: combination list
        """
        combinationList, flag = [], True
        for columnName, columnValues in list(structure.items())[:-1]:
            values = list(map(lambda x: columnName + '=>' + x, columnValues['values']))
            if flag:
                combinationList, flag = list(map(lambda x: [x], values)), False
            else:
                combinationList = self.addColumnValuesTolist(combinationList, values)
        return combinationList

    def addColumnValuesTolist(self, combinationList, values):
        """
        method to add add column values to list creating more sublist of combinations
        Attributes:
            combinationList (list): the list of combinations
            values(list): values to add to lists
        Returns:
            list: new combination list
        """
        newCombinationList = []
        for value in values:
            combinationCopy = copy.deepcopy(combinationList)
            for combination in combinationCopy:
                combination += [value]
            newCombinationList += combinationCopy
        return newCombinationList

    # test classification

    def classifyTest(self, testData, structure, rules):
        """
        method to classify test data by given classifier type
        Parameters:
                testData(list) : list of lines in files each element is a list
                structure(dict): the structure of data set returns {} if data set is empty, each element is
                            columnName : {'index': index , 'values': [values]} or
                            columnName : {'index': index , 'values': ["Numeric"]
                rules(list): list of rules
                classifierType(String): classifier type
        Returns:
            list: classified Data
        """
        newTestData = copy.deepcopy(testData)
        classIndex = structure['class']['index']
        for row in newTestData:
            row[classIndex] = self.testAttribute(row, structure, self.convertStringRulesToLists(rules))
        return newTestData

    def testAttribute(self, row, structure, rules):
        """
        method to classify a row
        Parameters:
            row(list): the row to classify
            structure(dict): the structure of data set returns {} if data set is empty, each element is
                columnName : {'index': index , 'values': [values]} or
                columnName : {'index': index , 'values': ["Numeric"]
            rules(list): list of list rules
        """
        for rule in rules:
            flag = True
            oddSpots = rule[1:-2:2]
            evenSpots = rule[0:-2:2]
            for i, j in zip(evenSpots, oddSpots):
                if row[structure[i]['index']] != j:
                    flag = False
                    break
            if flag:
                return rule[len(rule)-1].strip()

    def convertStringRulesToLists(self, rules):
        """
        method to convert a list with String rules to a list with list rules
        Parameters:
            rules(list): list of rules
        """
        newRules = []
        for rule in rules:
            rule = rule.replace(',', '==').replace('=>', '==').split('==')
            for i in range(0, len(rule)):
                rule[i] = rule[i].strip()
            newRules += [rule]
        return newRules

    def checkAccuracyOfClassifier(self, newData, oldData):
        """
        function to test accuracy of are Id3Classifier
        Parameters:
            newData(list): the test data we classified
            oldData(list): the test data with class attributes
        Returns:
            float: Accuracy percent
        """
        error = 0
        for i, j in zip(newData, oldData):
            if i != j:
                error += 1
        return ((len(newData) - error) / len(newData)) * 100 if len(newData) > 0 else 100

    def createNewStructureWithoutItem(self, structure, itemName):
        """
        method to create a copy of structure without an item
        Attributes:
                structure(dict): the structure of data set returns {} if data set is empty, each element is
                            columnName : {'index': index , 'values': [values]} or
                            columnName : {'index': index , 'values': ["Numeric"]
                itemName(String): the name of item to remove from structure
        Returns:
            dict: new structure without item
        """
        newStructure = copy.deepcopy(structure)
        del newStructure[itemName]
        return newStructure


class DecisionTree:
    def __init__(self, name, value, N=0, Nc=0):
        """"
        Ctor for DataLoader
        Attributes:
            name(string): the name of node
            value(string): the value of node
            N(int): the number of attributes in data at this node
            Nc(int): the number of attributes in data at this node with majority class value
        """
        self.name = name
        self.value = value
        self.SubDecisionTree = None
        self.N = N
        self.Nc = Nc

    def addSubDecisionTree(self, node):
        """
        method to add sons to a node
        Attributes:
            node (list): son to add to current node sons List
        """
        if self.SubDecisionTree is None:
            self.SubDecisionTree = []
        self.SubDecisionTree += node
