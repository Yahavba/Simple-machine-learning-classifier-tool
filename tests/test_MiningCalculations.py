import unittest
from MiningCalculations import MiningCalculator


class TestMiningCalculations(unittest.TestCase):
    calculator = None
    dataOne = []
    dataTwo = []
    structureOne = []
    structureTwo = []

    def setUp(self):
        self.calculator = MiningCalculator()
        self.dataOne = [["4", "no"], ["5", "yes"], ["8", "no"], ["12", "yes"], ["15", "yes"]]
        self.structureOne = {"Hours": {"index": 0, "values": ["Numeric"]},
                             "class": {"index": 1, "values": ["no", "yes"]}}
        self.dataTwo = [["M", "Diaspora", "Over 700", "High"], ["M", "Israel", "600-700", "Medium"],
                        ["F", "Israel", "600-700", "High"], ["F", "Diaspora", "0-600", "Medium"],
                        ["F", "Israel", "0-600", "Low"]]
        self.structureTwo = {"Gender": {"index": 0, "values": ["M", "F"]},
                             "Place": {"index": 1, "values": ["Diaspora", "Israel"]},
                             "Test": {"index": 2, "values": ["0-600", "600-700", "Over 700"]},
                             "class": {"index": 3, "values": ["Low", "Medium", "High"]}}

    def test_calcClassEntropy(self):
        self.assertEqual(0.971, self.calculator.calcDataEntropy(self.dataOne, self.structureOne))
        self.assertEqual(1.522, self.calculator.calcDataEntropy(self.dataTwo, self.structureTwo))

    def test_calcClassEntropyByColumn(self):
        self.assertEqual(1.351, self.calculator.calcEntropyByColumnSplit(self.dataTwo, self.structureTwo, "Gender"))
        self.assertEqual(1.351, self.calculator.calcEntropyByColumnSplit(self.dataTwo, self.structureTwo, "Place"))
        self.assertEqual(0.800, self.calculator.calcEntropyByColumnSplit(self.dataTwo, self.structureTwo, "Test"))

    def test_calcClassEntropyBySplit(self):
        self.assertEqual(0.649, self.calculator.calcEntropyBySplitValue(self.dataOne, self.structureOne, "Hours", 4.5))
        self.assertEqual(0.951, self.calculator.calcEntropyBySplitValue(self.dataOne, self.structureOne, "Hours", 6.5))
        self.assertEqual(0.551, self.calculator.calcEntropyBySplitValue(self.dataOne, self.structureOne, "Hours", 10))
        self.assertEqual(0.8, self.calculator.calcEntropyBySplitValue(self.dataOne, self.structureOne, "Hours", 13.5))

    def test_calcInfoGainByColumn(self):
        self.assertEqual(0.171, self.calculator.calcInfoGainByColumnSplit(self.dataTwo, self.structureTwo, "Gender"))
        self.assertEqual(0.171, self.calculator.calcInfoGainByColumnSplit(self.dataTwo, self.structureTwo, "Place"))
        self.assertEqual(0.722, self.calculator.calcInfoGainByColumnSplit(self.dataTwo, self.structureTwo, "Test"))

    def test_calcInfoGainBySplit(self):
        self.assertEqual(0.322, self.calculator.calcInfoGainBySplitValue(self.dataOne, self.structureOne, "Hours", 4.5))
        self.assertEqual(0.02, self.calculator.calcInfoGainBySplitValue(self.dataOne, self.structureOne, "Hours", 6.5))
        self.assertEqual(0.42, self.calculator.calcInfoGainBySplitValue(self.dataOne, self.structureOne, "Hours", 10))
        self.assertEqual(0.171, self.calculator.calcInfoGainBySplitValue(self.dataOne, self.structureOne, "Hours", 13.5))

    def test_findBestSplitInDataByInfoGain(self):
        self.assertEqual([10, 0.42],
                         self.calculator.findBestSplitInDataByInfoGain(self.dataOne, self.structureOne, "Hours"))

    def test_fillBestSplitsInDataByInfoGainIntoSplitList(self):
        splitList = {}

        self.calculator.fillBestSplitsInDataByInfoGainIntoDict(self.dataOne, self.structureOne,
                                                                    "Hours", 2, splitList, 0)
        self.assertEqual({"0": [[10.0, 0.42]], '1': [[6.5, 0.251], [13.5, 0.0]]}, splitList)

    def test_getBestSplitsInDataByInfoGain(self):
        self.assertEqual([10, 6.5],
                         self.calculator.getBestSplitsInDataByInfoGain(self.dataOne, self.structureOne, "Hours", 2))

        self.assertEqual([10, 6.5, 13.5, 4.5],
                         self.calculator.getBestSplitsInDataByInfoGain(self.dataOne, self.structureOne, "Hours", 5))

    def test_gini(self):
        self.assertEqual(0.48, self.calculator.calcDataGini(self.dataOne, self.structureOne))

    def test_giniSplitByValue(self):
        self.assertEqual(0.3, self.calculator.calcGiniSplitBySplitValue(self.dataOne, self.structureOne, 0, 4.5))
        self.assertEqual(0.466, self.calculator.calcGiniSplitBySplitValue(self.dataOne, self.structureOne, 0, 6.5))
        self.assertEqual(0.266, self.calculator.calcGiniSplitBySplitValue(self.dataOne, self.structureOne, 0, 10))
        self.assertEqual(0.4, self.calculator.calcGiniSplitBySplitValue(self.dataOne, self.structureOne, 0, 13.5))

    def test_findBestValueSplitByGini(self):
        self.assertEqual([10, 0.266], self.calculator.findBestValueSplitByGini(self.dataOne, self.structureOne, 0))

    def test_fillListWithDataValueSplitsByGini(self):
        splitsList = {}
        self.calculator.fillDictWithBestValueSplitsOfDataByGini(self.dataOne, self.structureOne, 0, 2, splitsList, 0)

        self.assertEqual({'0': [[10.0, 0.266]], '1': [[6.5, 0.333], [13.5, 0.0]]}, splitsList)

    def test_getListWithDataValueSplitsByGini(self):
        self.assertEqual([10.0, 13.5],
                         self.calculator.getListWithBestValueSplitsOfDataByGini(self.dataOne, self.structureOne, 0, 2))

    def test_giniSplitByColumn(self):
        self.assertEqual(0.6, self.calculator.calcGiniSplitByColumn(self.dataTwo, self.structureTwo, "Gender"))
        self.assertEqual(0.6, self.calculator.calcGiniSplitByColumn(self.dataTwo, self.structureTwo, "Place"))
        self.assertEqual(0.4, self.calculator.calcGiniSplitByColumn(self.dataTwo, self.structureTwo, "Test"))

    def test_findBestColumnSplitByGini(self):
        self.assertEqual("Test", self.calculator.findBestColumnSplitByGini(self.dataTwo, self.structureTwo))

    def test_calcGainRatioSplitByColumn(self):
        self.assertEqual(0.176, self.calculator.calcGainRatioSplitByColumn(self.dataTwo, self.structureTwo, "Gender"))
        self.assertEqual(0.176, self.calculator.calcGainRatioSplitByColumn(self.dataTwo, self.structureTwo, "Place"))
        self.assertEqual(0.474, self.calculator.calcGainRatioSplitByColumn(self.dataTwo, self.structureTwo, "Test"))

    def test_findBestColumnSplitByGainRatio(self):
        self.assertEqual("Test", self.calculator.findBestColumnSplitByGainRatio(self.dataTwo, self.structureTwo))

    def test_removeDuplicatesInList(self):
        data = [0, 0, 0, 1, 1, 1, 2, 2]

        self.calculator.removeDuplicatesInList(data)

        self.assertEqual([0, 1, 2], data)

    def test_mostCommonClassAttribute(self):
        mostCommon = self.calculator.mostCommonClassAttribute(self.dataOne, self.structureOne)

        self.assertEqual("yes", mostCommon)

    def test_allRowsWithSameClass(self):
        answer = self.calculator.allRowsWithSameClass(self.dataOne, self.structureOne)

        self.assertEqual(False, answer)

    def test_calcNumberOfMajorityClassRows(self):
        answer = self.calculator.calcNumberOfMajorityClassRows(self.dataOne, self.structureOne)

        self.assertEqual(3, answer)

    def test_getSplitFunc(self):
        answer1 = self.calculator.getSplitFunc("Info Gain")
        answer2 = self.calculator.getSplitFunc("Gini Index")
        answer3 = self.calculator.getSplitFunc("Gain Ratio")
        answer4 = self.calculator.getSplitFunc("afoowa")

        self.assertEqual(self.calculator.findBestColumnSplitByInfoGain, answer1)
        self.assertEqual(self.calculator.findBestColumnSplitByGini, answer2)
        self.assertEqual(self.calculator.findBestColumnSplitByGainRatio, answer3)
        self.assertEqual(None, answer4)

    def test_calcProbabilityOfValGivenClassWithLaplaceCorrection(self):
        answer = self.calculator.calcProbabilityOfValGivenClassWithLaplaceCorrection(self.dataTwo, 0, "M", "High", 2)

        self.assertEqual(0.286, answer)

    def test_calcProbabilityOfClassValueWithLaplaceCorrection(self):
        answer = self.calculator.calcProbabilityOfClassValueWithLaplaceCorrection(self.dataTwo, "High", 3)

        self.assertEqual(0.375, answer)


if __name__ == '__main__':
    unittest.main()
