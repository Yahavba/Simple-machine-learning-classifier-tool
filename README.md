# Simple machine learning classifier tool

A Machine Learning classifier desktop application.

This Project was created only with <b> Python</b>. 

All algorithms and task are implemented from zero.

Tool includes:

1. Data cleaning.

2. Data discretization - discretization of data by one of the fiven techniwues (Entropy, Equal width, Equal depth or Gini index) into a given number of  bins.

3. Classifier building - building a selected classifier (ID3 or Naive Bayes).

Given Data set needs to be a .csv file with data columns and last column should be with the name class (the column we want to classify rows by its values).
if the file is not in this structure it will not work.
There is two data samples in the "Data" folder.

Each step creats a .csv file with the result of the step and at the end there is a .txt file with classification rules (each row is a rule).


### Project Setup (Windows):

1. make sure you have python on your computer (if not install python 3.6.1 from here [Python download](https://www.python.org/downloads/windows/))
2. Make sure Python is in path (if not follow this guide [Add python to path](https://datatofish.com/add-python-to-windows-path/))
3. Make sure pip is in path (if not follow this guide [Add pip to path](https://appuals.com/fix-pip-is-not-recognized-as-an-internal-or-external-command/))
5. Clone repository.
6. Run installationWin.bat and wait until console closes.
7. That’s it, you are all set up to run.

### Project Setup (Linux):

1. make sure you have python on your computer (if not install python from here [Python download](https://docs.python-guide.org/starting/install3/linux/))
3. Make sure you have pip on your computer (if not follow this guide [pip download]](https://itsfoss.com/install-pip-ubuntu/))
5. Clone repository.
6. Run installationLinux.sh and wait until console closes.
7. That’s it, you are all set up to run.

### Project Run:

#### Train:
In the console run:
1. Run runWin.bat (in Devops Scripts folder).

#### Test:
In the console run:
1. Run runLinux.sh (in Devops Scripts folder).

### Demo

![Simple Machine Learning Classifier Demo Video](https://github.com/leorrose/Simple-machine-learning-classifier-tool/blob/master/demo.gif)

Please let me know if you find bugs or something that needs to be fixed.

Hope you enjoy.
