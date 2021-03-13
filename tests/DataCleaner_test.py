import unittest
from DataCleaner import Cleaner


class TestDataCleaner(unittest.TestCase):
    dataCleaner = None
    dataWithNoCleaningNeeded = []
    dataWithMissingClass = []
    dataWithMissingValues = []
    dataWithMissingValuesAndClass = []
    structure = {}

    def setUp(self):
        self.dataCleaner = Cleaner()
        self.dataWithNoCleaningNeeded = [["13", "1000", "yes"], ["18", "5000", "no"], ["15", "3000", "no"],
                                         ["14", "800", "yes"]]
        self.dataWithMissingClass = [["13", "1000", "yes"], ["18", "5000", ""], ["15", "3000", "no"], ["14", "800", ""]]
        self.dataWithMissingValues = [["13", "", "yes"], ["", "5000", "no"], ["15", "3000", "no"]]
        self.dataWithMissingValuesAndClass = [["13", "", "yes"], ["18", "5000", ""], ["", "3000", "no"]]
        self.structure = {"Age": {"index": 0, "values": ["Numeric"]}, "Income": {"index": 1, "values": ["Numeric"]},
                          "class": {"index": 2, "values": ["yes", "no"]}}

    def test_cleanTrainingSet_dataWithNoCleaningNeeded(self):
        self.dataCleaner.cleanTrainingSet(self.dataWithNoCleaningNeeded, self.structure)

        self.assertEqual([["13", "1000", "yes"], ["18", "5000", "no"], ["15", "3000", "no"], ["14", "800", "yes"]],
                         self.dataWithNoCleaningNeeded)

    def test_cleanTrainingSet_dataWithMissingClass(self):
        self.dataCleaner.cleanTrainingSet(self.dataWithMissingClass, self.structure)

        self.assertEqual([["13", "1000", "yes"], ["15", "3000", "no"]], self.dataWithMissingClass)

    def test_cleanTrainingSet_dataWithMissingValues(self):
        self.dataCleaner.cleanTrainingSet(self.dataWithMissingValues, self.structure)

        self.assertEqual([["13", "4000.0", "yes"], ["15.0", "5000", "no"], ["15", "3000", "no"]], self.dataWithMissingValues)

    def test_cleanTrainingSet_dataWithMissingValuesAndClass(self):
        self.dataCleaner.cleanTrainingSet(self.dataWithMissingValuesAndClass, self.structure)

        self.assertEqual([["13", "3000.0", "yes"], ["13.0", "3000", "no"]], self.dataWithMissingValuesAndClass)

    def test_cleanTestSet_dataWithNoCleaningNeeded(self):
        self.dataCleaner.cleanTestSet(self.dataWithNoCleaningNeeded, self.structure)

        self.assertEqual([["13", "1000", "yes"], ["18", "5000", "no"], ["15", "3000", "no"], ["14", "800", "yes"]],
                         self.dataWithNoCleaningNeeded)

    def test_cleanTestSet_dataWithMissingClass(self):
        self.dataCleaner.cleanTestSet(self.dataWithMissingClass, self.structure)
        self.assertEqual([["13", "1000", "yes"], ["18", "5000", ""], ["15", "3000", "no"], ["14", "800", ""]],
                         self.dataWithMissingClass)

    def test_cleanTestSet_dataWithMissingValues(self):
        self.dataCleaner.cleanTestSet(self.dataWithMissingValues, self.structure)

        self.assertEqual([["13", "4000.0", "yes"], ["15.0", "5000", "no"], ["15", "3000", "no"]],
                         self.dataWithMissingValues)

    def test_cleanTestSet_dataWithMissingValuesAndClass(self):
        self.dataCleaner.cleanTestSet(self.dataWithMissingValuesAndClass, self.structure)
        self.assertEqual([["13", "4000.0", "yes"], ["18", "5000", ""], ["15.5", "3000", "no"]],
                         self.dataWithMissingValuesAndClass)

    def test_removeRows_dataWithNoCleaningNeeded(self):
        self.dataCleaner.removeRows(self.dataWithNoCleaningNeeded, self.structure)

        self.assertEqual([["13", "1000", "yes"], ["18", "5000", "no"], ["15", "3000", "no"], ["14", "800", "yes"]],
                         self.dataWithNoCleaningNeeded)

    def test_removeRows_dataWithMissingClass(self):
        self.dataCleaner.removeRows(self.dataWithMissingClass, self.structure)
        self.assertEqual([["13", "1000", "yes"], ["15", "3000", "no"]], self.dataWithMissingClass)

    def test_removeRows_dataWithMissingValues(self):
        self.dataCleaner.removeRows(self.dataWithMissingValues, self.structure)

        self.assertEqual([["13", "", "yes"], ["", "5000", "no"], ["15", "3000", "no"]],
                         self.dataWithMissingValues)

    def test_removeRows_dataWithMissingValuesAndClass(self):
        self.dataCleaner.removeRows(self.dataWithMissingValuesAndClass, self.structure)
        self.assertEqual([["13", "", "yes"], ["", "3000", "no"]], self.dataWithMissingValuesAndClass)

    def test_fillMissingValues_dataWithNoCleaningNeeded(self):
        self.dataCleaner.fillMissingValues(self.dataWithNoCleaningNeeded, self.structure)

        self.assertEqual([["13", "1000", "yes"], ["18", "5000", "no"], ["15", "3000", "no"], ["14", "800", "yes"]],
                         self.dataWithNoCleaningNeeded)

    def test_fillMissingValues_dataWithMissingClass(self):
        self.dataCleaner.fillMissingValues(self.dataWithMissingClass, self.structure)
        self.assertEqual([["13", "1000", "yes"], ["18", "5000", ""], ["15", "3000", "no"], ["14", "800", ""]],
                         self.dataWithMissingClass)

    def test_fillMissingValues_dataWithMissingValues(self):
        self.dataCleaner.fillMissingValues(self.dataWithMissingValues, self.structure)

        self.assertEqual([["13", "4000.0", "yes"], ["15.0", "5000", "no"], ["15", "3000", "no"]],
                         self.dataWithMissingValues)

    def test_fillMissingValues_dataWithMissingValuesAndClass(self):
        self.dataCleaner.fillMissingValues(self.dataWithMissingValuesAndClass, self.structure)
        self.assertEqual([["13", "4000.0", "yes"], ["18", "5000", ""], ["15.5", "3000", "no"]],
                         self.dataWithMissingValuesAndClass)

    def test_fillNumericValuesInColumn_dataWithNoCleaningNeeded(self):
        self.dataCleaner.fillNumericValuesInColumn(self.dataWithNoCleaningNeeded, self.structure, 0)
        self.dataCleaner.fillNumericValuesInColumn(self.dataWithNoCleaningNeeded, self.structure, 1)

        self.assertEqual([["13", "1000", "yes"], ["18", "5000", "no"], ["15", "3000", "no"], ["14", "800", "yes"]],
                         self.dataWithNoCleaningNeeded)

    def test_fillNumericValuesInColumn_dataWithMissingClass(self):
        self.dataCleaner.fillNumericValuesInColumn(self.dataWithMissingClass, self.structure, 0)
        self.dataCleaner.fillNumericValuesInColumn(self.dataWithMissingClass, self.structure, 1)

        self.assertEqual([["13", "1000", "yes"], ["18", "5000", ""], ["15", "3000", "no"], ["14", "800", ""]],
                         self.dataWithMissingClass)

    def test_fillNumericValuesInColumn_dataWithMissingValues(self):
        self.dataCleaner.fillNumericValuesInColumn(self.dataWithMissingValues, self.structure, 0)
        self.dataCleaner.fillNumericValuesInColumn(self.dataWithMissingValues, self.structure, 1)

        self.assertEqual([["13", "4000.0", "yes"], ["15.0", "5000", "no"], ["15", "3000", "no"]],
                         self.dataWithMissingValues)

    def test_fillNumericValuesInColumn_dataWithMissingValuesAndClass(self):
        self.dataCleaner.fillNumericValuesInColumn(self.dataWithMissingValuesAndClass, self.structure, 0)
        self.dataCleaner.fillNumericValuesInColumn(self.dataWithMissingValuesAndClass, self.structure, 1)

        self.assertEqual([["13", "4000.0", "yes"], ["18", "5000", ""], ["15.5", "3000", "no"]],
                         self.dataWithMissingValuesAndClass)

    def test_AverageListByClass_dataWithNoCleaningNeeded(self):
        self.assertEqual(["13.5", "16.5"],
                         self.dataCleaner.AverageListByClass(self.dataWithNoCleaningNeeded, self.structure, 0))
        self.assertEqual(["900.0", "4000.0"],
                         self.dataCleaner.AverageListByClass(self.dataWithNoCleaningNeeded, self.structure, 1))

    def test_AverageListByClass_dataWithMissingClass(self):
        self.assertEqual(["13.0", "15.0"],
                         self.dataCleaner.AverageListByClass(self.dataWithMissingClass, self.structure, 0))
        self.assertEqual(["1000.0", "3000.0"],
                         self.dataCleaner.AverageListByClass(self.dataWithMissingClass, self.structure, 1))

    def test_AverageListByClass_dataWithMissingValues(self):
        self.assertEqual(["13.0", "15.0"],
                         self.dataCleaner.AverageListByClass(self.dataWithMissingValues, self.structure, 0))
        self.assertEqual([None, "4000.0"],
                         self.dataCleaner.AverageListByClass(self.dataWithMissingValues, self.structure, 1))

    def test_AverageListByClass_dataWithMissingValuesAndClass(self):
        self.assertEqual(["13.0", None],
                         self.dataCleaner.AverageListByClass(self.dataWithMissingValuesAndClass, self.structure, 0))
        self.assertEqual([None, "3000.0"],
                         self.dataCleaner.AverageListByClass(self.dataWithMissingValuesAndClass, self.structure, 1))

    def test_fillCategorialValuesInColumn_dataWithNoCleaningNeeded(self):
        self.structure = {"gender": {"index": 0, "values": ["male", "female"]},
                          "class": {"index": 1, "values": ["no", "yes"]}}
        data = [["male", "no"], ["female", "no"], ["female", "yes"], ["male", "yes"]]

        self.dataCleaner.fillCategorialValuesInColumn(data, self.structure, 0)

        self.assertEqual([["male", "no"], ["female", "no"], ["female", "yes"], ["male", "yes"]], data)

    def test_fillCategorialValuesInColumn_dataWithMissingClass(self):
        self.structure = {"gender": {"index": 0, "values": ["male", "female"]},
                          "class": {"index": 1, "values": ["no", "yes"]}}
        data = [["male", ""], ["female", "no"], ["female", ""], ["male", "yes"]]

        self.dataCleaner.fillCategorialValuesInColumn(data, self.structure, 0)

        self.assertEqual([["male", ""], ["female", "no"], ["female", ""], ["male", "yes"]], data)

    def test_fillCategorialValuesInColumn_dataWithMissingValues(self):
        self.structure = {"gender": {"index": 0, "values": ["male", "female"]},
                          "class": {"index": 1, "values": ["no", "yes"]}}
        data = [["", "no"], ["female", "no"], ["", "yes"], ["male", "yes"]]

        self.dataCleaner.fillCategorialValuesInColumn(data, self.structure, 0)

        self.assertEqual([["female", "no"], ["female", "no"], ["male", "yes"], ["male", "yes"]], data)

    def test_fillCategorialValuesInColumn_dataWithMissingValuesAndClass(self):
        self.structure = {"gender": {"index": 0, "values": ["male", "female"]},
                          "class": {"index": 1, "values": ["no", "yes"]}}
        data = [["", "no"], ["female", "no"], ["", "yes"], ["male", "yes"], ["male", ""]]

        self.dataCleaner.fillCategorialValuesInColumn(data, self.structure, 0)

        self.assertEqual([["female", "no"], ["female", "no"], ["male", "yes"], ["male", "yes"], ["male", ""]], data)

    def test_commonValuesByClass_dataWithNoCleaningNeeded(self):
        self.structure = {"gender": {"index": 0, "values": ["male", "female"]},
                          "class": {"index": 1, "values": ["no", "yes"]}}
        data = [["male", "no"], ["male", "no"], ["female", "yes"], ["female", "yes"]]

        self.assertEqual(["male", "female"], self.dataCleaner.commonValuesByClass(data, self.structure, 0))

    def test_commonValuesByClass_dataWithMissingClass(self):
        self.structure = {"gender": {"index": 0, "values": ["male", "female"]},
                          "class": {"index": 1, "values": ["no", "yes"]}}
        data = [["male", ""], ["female", "no"], ["female", ""], ["male", "yes"]]

        self.assertEqual(["female", "male"], self.dataCleaner.commonValuesByClass(data, self.structure, 0))

    def test_commonValuesByClass_dataWithMissingValues(self):
        self.structure = {"gender": {"index": 0, "values": ["male", "female"]},
                          "class": {"index": 1, "values": ["no", "yes"]}}
        data = [["", "no"], ["female", "no"], ["", "yes"], ["male", "yes"]]

        self.assertEqual(["female", "male"], self.dataCleaner.commonValuesByClass(data, self.structure, 0))

    def test_commonValuesByClass_dataWithMissingValuesAndClass(self):
        self.structure = {"gender": {"index": 0, "values": ["male", "female"]},
                          "class": {"index": 1, "values": ["no", "yes"]}}
        data = [["", "no"], ["female", "no"], ["", "yes"], ["male", "yes"], ["male", ""]]

        self.assertEqual(["female", "male"], self.dataCleaner.commonValuesByClass(data, self.structure, 0))

    def test_mostFrequentElement_fullData(self):
        data = ["male", "female", "male", "male", "male", "female"]

        self.assertEqual("male", self.dataCleaner.mostFrequentElement(data))

    def test_mostFrequentElement_dataWithMissingValues(self):
        data = ["male", "", "male", "male", "", "female"]

        self.assertEqual("male", self.dataCleaner.mostFrequentElement(data))


if __name__ == '__main__':
    unittest.main()