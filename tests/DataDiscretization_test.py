import unittest
from DataDiscretization import Discretization


class TestDataDiscretization(unittest.TestCase):
    discretization = None
    data = []
    dataTwo = []
    structure = []
    StructureTwo = []

    def setUp(self):
        self.discretization = Discretization()
        self.data = [["4", "no"], ["5", "yes"], ["8", "no"], ["12", "yes"], ["15", "yes"]]
        self.structure = {"Hours": {"index": 0, "values": ["Numeric"]},
                          "class": {"index": 1, "values": ["no", "yes"]}}
        self.dataTwo = [["M", "Diaspora", "730", "High"], ["M", "Israel", "680", "Medium"],
                        ["F", "Israel", "640", "High"], ["F", "Diaspora", "585", "Medium"],
                        ["F", "Israel", "570", "Low"]]
        self.structureTwo = {"Gender": {"index": 0, "values": ["M", "F"]},
                             "Place": {"index": 1, "values": ["Diaspora", "Israel"]},
                             "Test": {"index": 2, "values": ["Numeric"]},
                             "class": {"index": 3, "values": ["Low", "Medium", "High"]}}

    def test_discretizationData_ByEntropy(self):
        self.discretization.discretizationData(self.data[0:3], self.data[3:], self.structure, 2, "entropy")
        self.discretization.discretizationData(self.dataTwo, [], self.structureTwo, 2, "entropy")

        self.assertEqual([["value<=6.5", "no"], ["value<=6.5", "yes"], ["value>6.5", "no"],
                          ["value>6.5", "yes"], ["value>6.5", "yes"]], self.data)

        self.assertEqual([["F", "Israel", "value<=577.5", "Low"], ["F", "Diaspora", "value>577.5", "Medium"],
                          ["F", "Israel", "value>577.5", "High"], ["M", "Israel", "value>577.5", "Medium"],
                          ["M", "Diaspora", "value>577.5", "High"]], self.dataTwo)

    def test_discretizationData_ByEqualDepth(self):
        self.discretization.discretizationData(self.data, [], self.structure, 2, "Equal Depth")
        self.discretization.discretizationData(self.dataTwo, [], self.structureTwo, 2, "Equal Depth")

        self.assertEqual([["value<=8.0", "no"], ["value<=8.0", "yes"], ["value<=8.0", "no"],
                          ["value>8.0", "yes"], ["value>8.0", "yes"]], self.data)

        self.assertEqual([["F", "Israel", "value<=640.0", "Low"], ["F", "Diaspora", "value<=640.0", "Medium"],
                          ["F", "Israel", "value<=640.0", "High"], ["M", "Israel", "value>640.0", "Medium"],
                          ["M", "Diaspora", "value>640.0", "High"]], self.dataTwo)

    def test_discretizationData_ByGini(self):
        self.discretization.discretizationData(self.data, [], self.structure, 2, "Gini index")
        self.discretization.discretizationData(self.dataTwo, [], self.structureTwo, 2, "Gini index")

        self.assertEqual([["value<=10.0", "no"], ["value<=10.0", "yes"], ["value<=10.0", "no"],
                          ["value>10.0", "yes"], ["value>10.0", "yes"]], self.data)

        self.assertEqual([["F", "Israel", "value<=577.5", "Low"], ["F", "Diaspora", "value>577.5", "Medium"],
                          ["F", "Israel", "value>577.5", "High"], ["M", "Israel", "value>577.5", "Medium"],
                          ["M", "Diaspora", "value>577.5", "High"]], self.dataTwo)

    def test_discretizationData_ByEqualWidth(self):
        self.discretization.discretizationData(self.data, [], self.structure, 2, "Equal Width")
        self.discretization.discretizationData(self.dataTwo, [], self.structureTwo, 2, "Equal Width")

        self.assertEqual([["value<=5.5", "no"], ["value<=5.5", "yes"], ["value>5.5", "no"],
                          ["value>5.5", "yes"], ["value>5.5", "yes"]], self.data)

        self.assertEqual([["F", "Israel", "value>80.0", "Low"], ["F", "Diaspora", "value>80.0", "Medium"],
                          ["F", "Israel", "value>80.0", "High"], ["M", "Israel", "value>80.0", "Medium"],
                          ["M", "Diaspora", "value>80.0", "High"]], self.dataTwo)

    def test_discretizationOFDataByColumn(self):
        self.discretization.discretizationOFDataByColumn(self.data, 0, {"value<=10.0": lambda x: x <= 10, "value>10.0": lambda x: x > 10})

        self.discretization.discretizationOFDataByColumn(self.dataTwo, 2,
                                                         {"value<=577.5": lambda x: x <= 577.5, "value>577.5": lambda x: x > 577.5})

        self.assertEqual([["value<=10.0", "no"], ["value<=10.0", "yes"], ["value<=10.0", "no"],
                          ["value>10.0", "yes"], ["value>10.0", "yes"]], self.data)

        self.assertEqual([["M", "Diaspora", "value>577.5", "High"], ["M", "Israel", "value>577.5", "Medium"],
                          ["F", "Israel", "value>577.5", "High"], ["F", "Diaspora", "value>577.5", "Medium"],
                          ["F", "Israel", "value<=577.5", "Low"]], self.dataTwo)

    def test_sortDataByAscendingOrderOFValuesInColumn(self):
        self.data.reverse()

        self.discretization.sortDataByAscendingOrderOFValuesInColumn(self.data, 0)

        self.assertEqual([["4", "no"], ["5", "yes"], ["8", "no"], ["12", "yes"], ["15", "yes"]], self.data)

    def test_createBinsByEqualWidth(self):
        bins = self.discretization.createBinsByEqualWidth(self.data, self.structure['Hours']['index'], 2)
        binsTwo = self.discretization.createBinsByEqualWidth(self.data, self.structure['Hours']['index'], 3)

        self.assertEqual(["value<=5.5", "value>5.5"], list(bins.keys()))
        self.assertEqual(["value<=3.667", "3.667<value<=7.334", "value>7.334"], list(binsTwo.keys()))

    def test_createBinsByEqualDepth(self):
        bins = self.discretization.createBinsByEqualDepth(self.data, self.structure['Hours']['index'], 2)
        binsTwo = self.discretization.createBinsByEqualDepth(self.data, self.structure['Hours']['index'], 3)

        self.assertEqual(["value<=8.0", "value>8.0"], list(bins.keys()))
        self.assertEqual(["value<=5.0", "5.0<value<=12.0", "value>12.0"], list(binsTwo.keys()))

    def test_createBinsByEntropy(self):
        bins = self.discretization.createBinsByEntropy(self.data, self.structure, 'Hours', 2)
        binsTwo = self.discretization.createBinsByEntropy(self.data, self.structure, 'Hours', 3)

        self.assertEqual(["value<=10.0", "value>10.0"], list(bins.keys()))
        self.assertEqual(["value<=6.5", "6.5<value<=10.0", "value>10.0"], list(binsTwo.keys()))

    def test_createBinsByGiniIndex(self):
        bins = self.discretization.createBinsByGiniIndex(self.data, self.structure, 0, 2)
        binsTwo = self.discretization.createBinsByGiniIndex(self.data, self.structure, 0, 3)

        self.assertEqual(["value<=10.0", "value>10.0"], list(bins.keys()))
        self.assertEqual(["value<=10.0", "10.0<value<=13.5", "value>13.5"], list(binsTwo.keys()))


if __name__ == '__main__':
    unittest.main()
