import unittest
from DataClassifier import Classifier
from MiningCalculations import MiningCalculator


class TestDataClassifier(unittest.TestCase):
    calculator = None
    data = []
    structure = []

    def setUp(self):
        self.classifier = Classifier()
        self.calculator = MiningCalculator()
        self.data = [["M", "Diaspora", "Over 700", "High"], ["M", "Israel", "600-700", "Medium"],
                     ["F", "Israel", "600-700", "High"], ["F", "Diaspora", "0-600", "Medium"],
                     ["F", "Israel", "0-600", "Low"]]
        self.structure = {"Gender": {"index": 0, "values": ["M", "F"]},
                          "Place": {"index": 1, "values": ["Diaspora", "Israel"]},
                          "Test": {"index": 2, "values": ["0-600", "600-700", "Over 700"]},
                          "class": {"index": 3, "values": ["Low", "Medium", "High"]}}

    def test_buildClassifier_ID3(self):
        rules = self.classifier.buildClassifier(self.data, self.structure, "id3", self.calculator.findBestColumnSplitByInfoGain)

        self.assertEqual(['Test == 0-600 , Place == Diaspora => class == Medium', 'Test == 0-600 , Place == Israel => class == Low',
                          'Test == 600-700 , Gender == M => class == Medium', 'Test == 600-700 , Gender == F => class == High',
                          'Test == Over 700 => class == High'], rules)

    def test_buildClassifier_NaiveBayes(self):
        rules = self.classifier.buildClassifier(self.data, self.structure, "Naive Bayes")

        self.assertEqual(["Gender == M, Place == Diaspora, Test == 0-600 => class == Medium",
         "Gender == F, Place == Diaspora, Test == 0-600 => class == Medium",
         "Gender == M, Place == Israel, Test == 0-600 => class == Medium",
         "Gender == F, Place == Israel, Test == 0-600 => class == Medium",
         "Gender == M, Place == Diaspora, Test == 600-700 => class == High",
         "Gender == F, Place == Diaspora, Test == 600-700 => class == High",
         "Gender == M, Place == Israel, Test == 600-700 => class == High",
         "Gender == F, Place == Israel, Test == 600-700 => class == High",
         "Gender == M, Place == Diaspora, Test == Over 700 => class == High",
         "Gender == F, Place == Diaspora, Test == Over 700 => class == High",
         "Gender == M, Place == Israel, Test == Over 700 => class == High",
         "Gender == F, Place == Israel, Test == Over 700 => class == High"], rules)

    def test_buildId3Classifier(self):
        rules = self.classifier.buildId3Classifier(self.data, self.structure, "Medium", self.calculator.findBestColumnSplitByInfoGain)

        self.assertEqual(['Test == 0-600 , Place == Diaspora => class == Medium', 'Test == 0-600 , Place == Israel => class == Low',
                          'Test == 600-700 , Gender == M => class == Medium', 'Test == 600-700 , Gender == F => class == High',
                          'Test == Over 700 => class == High'], rules)

    def test_convertStringRulesToLists(self):
        rules = self.classifier.buildId3Classifier(self.data, self.structure, "Medium", self.calculator.findBestColumnSplitByInfoGain)

        rules = self.classifier.convertStringRulesToLists(rules)

        self.assertEqual([['Test', '0-600', 'Place', 'Diaspora', 'class', 'Medium'],
                          ['Test', '0-600', 'Place', 'Israel', 'class', 'Low'],
                          ['Test', '600-700', 'Gender', 'M', 'class', 'Medium'],
                          ['Test', '600-700', 'Gender', 'F', 'class', 'High'],
                          ['Test', 'Over 700', 'class', 'High']], rules)

    def test_testAttribute(self):
        rules = self.classifier.buildId3Classifier(self.data, self.structure, "Medium", self.calculator.findBestColumnSplitByInfoGain)
        rules = self.classifier.convertStringRulesToLists(rules)

        self.assertEqual("Medium",
                         self.classifier.testAttribute(["M", "Israel", "600-700", "Medium"], self.structure, rules))

    def test_classifyTest_ID3(self):
        rules = self.classifier.buildId3Classifier(self.data, self.structure, "Medium", self.calculator.findBestColumnSplitByInfoGain)

        self.assertEqual(self.data,
                         self.classifier.classifyTest(self.data, self.structure, rules))

    def test_classifyTest_NaiveBayes(self):
        rules = self.classifier.buildNaiveBayesClassifier(self.structure, self.data)

        self.assertEqual([['M', 'Diaspora', 'Over 700', 'High'], ['M', 'Israel', '600-700', 'High'],
                          ['F', 'Israel', '600-700', 'High'], ['F', 'Diaspora', '0-600', 'Medium'],
                          ['F', 'Israel', '0-600', 'Medium']],
                         self.classifier.classifyTest(self.data, self.structure, rules))

    def test_checkAccuracyOfClassifier_Id3(self):
        rules = self.classifier.buildId3Classifier(self.data, self.structure, "Medium", self.calculator.findBestColumnSplitByInfoGain)
        newData = self.classifier.classifyTest(self.data, self.structure, rules)

        self.assertEqual(100, self.classifier.checkAccuracyOfClassifier(newData, self.data))

    def test_checkAccuracyOfClassifier_NaiveBayes(self):
        rules = self.classifier.buildNaiveBayesClassifier(self.structure, self.data)
        newData = self.classifier.classifyTest(self.data, self.structure, rules)

        self.assertEqual(60, self.classifier.checkAccuracyOfClassifier(newData, self.data))

    def test_buildNaiveBayesClassifier(self):
        rules = self.classifier.buildNaiveBayesClassifier(self.structure, self.data)

        self.assertEqual(["Gender == M, Place == Diaspora, Test == 0-600 => class == Medium",
         "Gender == F, Place == Diaspora, Test == 0-600 => class == Medium",
         "Gender == M, Place == Israel, Test == 0-600 => class == Medium",
         "Gender == F, Place == Israel, Test == 0-600 => class == Medium",
         "Gender == M, Place == Diaspora, Test == 600-700 => class == High",
         "Gender == F, Place == Diaspora, Test == 600-700 => class == High",
         "Gender == M, Place == Israel, Test == 600-700 => class == High",
         "Gender == F, Place == Israel, Test == 600-700 => class == High",
         "Gender == M, Place == Diaspora, Test == Over 700 => class == High",
         "Gender == F, Place == Diaspora, Test == Over 700 => class == High",
         "Gender == M, Place == Israel, Test == Over 700 => class == High",
         "Gender == F, Place == Israel, Test == Over 700 => class == High"], rules)

    def test_classOfCombination(self):
        ProbabilityList = self.classifier.createProbabilityDict(self.structure, self.data)
        classValue = self.classifier.classOfCombination(self.data, ["Gender=>F", "Place=>Diaspora", "Test=>0-600"],
                                                                             ProbabilityList, ['Low', 'Medium', 'High'])

        self.assertEqual("Medium", classValue)

    def test_calcProbabilityOfCombinationGivenClass(self):
        ProbabilityList = self.classifier.createProbabilityDict(self.structure, self.data)
        probability = self.classifier.calcProbabilityOfCombinationGivenClass(["Gender=>F", "Place=>Diaspora", "Test=>0-600"],
                                                                             ProbabilityList, "Medium")

        self.assertEqual(0.02, probability)

    def test_createProbabilityList(self):
        probabilityDict = self.classifier.createProbabilityDict(self.structure, self.data)

        self.assertEqual({'High': {'Gender=>F': 0.286, 'Gender=>M': 0.286,
                                   'Place=>Diaspora': 0.286, 'Place=>Israel': 0.286,
                                   'Test=>0-600': 0.125, 'Test=>600-700': 0.25, 'Test=>Over 700': 0.25},
                          'Low': {'Gender=>F': 0.286, 'Gender=>M': 0.143,
                                  'Place=>Diaspora': 0.143, 'Place=>Israel': 0.286, 'Test=>0-600': 0.25,
                                  'Test=>600-700': 0.125, 'Test=>Over 700': 0.125},
                          'Medium': {'Gender=>F': 0.286, 'Gender=>M': 0.286, 'Place=>Diaspora': 0.286,
                                     'Place=>Israel': 0.286, 'Test=>0-600': 0.25, 'Test=>600-700': 0.25,
                                     'Test=>Over 700': 0.125}}, probabilityDict)

    def test_createColumnValuesCombination(self):
        combinations = self.classifier.createColumnValuesCombination(self.structure)

        self.assertEqual([['Gender=>M', 'Place=>Diaspora', 'Test=>0-600'], ['Gender=>F', 'Place=>Diaspora', 'Test=>0-600'],
                          ['Gender=>M', 'Place=>Israel', 'Test=>0-600'], ['Gender=>F', 'Place=>Israel', 'Test=>0-600'],
                          ['Gender=>M', 'Place=>Diaspora', 'Test=>600-700'], ['Gender=>F', 'Place=>Diaspora', 'Test=>600-700'],
                          ['Gender=>M', 'Place=>Israel', 'Test=>600-700'], ['Gender=>F', 'Place=>Israel', 'Test=>600-700'],
                          ['Gender=>M', 'Place=>Diaspora', 'Test=>Over 700'], ['Gender=>F', 'Place=>Diaspora', 'Test=>Over 700'],
                          ['Gender=>M', 'Place=>Israel', 'Test=>Over 700'], ['Gender=>F', 'Place=>Israel', 'Test=>Over 700']], combinations)

    def test_addColumnValuesTolist(self):
        combinations = self.classifier.addColumnValuesTolist([["M"], ["F"]], ["Diaspora", "Israel"])

        self.assertEqual([['M', 'Diaspora'], ['F', 'Diaspora'], ['M', 'Israel'], ['F', 'Israel']], combinations)


if __name__ == '__main__':
    unittest.main()

