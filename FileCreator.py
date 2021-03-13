import csv


class CreateFile:
    """ class for creating files in the process"""
    def __init__(self):
        pass

    def createCsvFile(self, structure, rows, name, pathToCreateFile):
        """
        method to create a csv file of columns from structure and lines from rows
        Attributes:
            pathToCreateFile(string): the path to create inside it a file
            name(String): the name of the file
            rows(list) : list of lines in files each element is a list
            structure(dict): the structure of data set returns {} if data set is empty, each element is
                            columnName : {'index': index , 'values': [values]} or
                            columnName : {'index': index , 'values': ["Numeric"]
        Raise:
            EnvironmentError
        """
        with open(pathToCreateFile + '/' + name + ".csv", "w", newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow(list(structure.keys()))
            csv_writer.writerows(rows)

    def createTxtFile(self, rows, name, pathToCreateFile):
        """
        method to create a txt file from rows
        Attributes:
            pathToCreateFile(string): the path to create inside it a file
            name(String): the name of the file
            rows(list) : list of lines in files each element is a list
        Raise:
            EnvironmentError
        """
        with open(pathToCreateFile + '/' + name + ".txt", "w") as txt_file:
            for row in rows:
                txt_file.write(row + '\n')
