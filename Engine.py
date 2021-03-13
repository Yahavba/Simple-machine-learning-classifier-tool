import tkinter
from tkinter import ttk, filedialog, messagebox
from ProcessController import BuildClassifierProcess
import threading


class Gui:
    currentFrame = None

    def __init__(self):
        """"
        Ctor for Gui
        """
        self.root = tkinter.Tk()
        self.root.config(background="lightblue")
        self.root.geometry('+{0}+{1}'.format(int((self.root.winfo_screenwidth()/2) - (self.root.winfo_reqwidth()/2)),
                                             int((self.root.winfo_screenheight()/2) - (self.root.winfo_reqheight()/2))))
        self.buildBuildingClassifierFrame()
        self.root.mainloop()

    def buildBuildingClassifierFrame(self):
        """"
        method to build frame of building classifier
        """
        self.root.geometry("500x350")
        self.root.title("Build Classifier Screen")
        self.root.resizable(False, False)

        if self.currentFrame:
            self.currentFrame.destroy()

        frame = tkinter.Frame(self.root, bg="lightblue")
        frame.pack(side=tkinter.TOP, fill=tkinter.X, padx=5, pady=5)

        firstFrame = tkinter.Frame(frame, bg="lightblue")
        firstFrame.pack(side=tkinter.TOP, fill=tkinter.X, padx=5, pady=5)
        classifierTypeLB = tkinter.Label(firstFrame, text="Classifier type: ", height=1, width=23, bg="lightblue")
        classifierTypeLB.pack(side=tkinter.LEFT, fill=tkinter.X, padx=5, pady=5)
        self.classifierTypeCB = ttk.Combobox(firstFrame, values=["ID3", "Naive Bayes"], state='readonly')
        self.classifierTypeCB.current(0)
        self.classifierTypeCB.bind("<<ComboboxSelected>>", self.lockComboBoxForNaiveByase)
        self.classifierTypeCB.pack(side=tkinter.LEFT, fill=tkinter.X, padx=5, pady=5)

        secondFrame = tkinter.Frame(frame, bg="lightblue")
        secondFrame.pack(side=tkinter.TOP, fill=tkinter.X, padx=5, pady=5)
        classifierSplitTypeLB = tkinter.Label(secondFrame, text="Classifier split type: ", height=1, width=23, bg="lightblue")
        classifierSplitTypeLB.pack(side=tkinter.LEFT, fill=tkinter.X, padx=5, pady=5)
        self.classifierSplitTypeCB = ttk.Combobox(secondFrame, values=["Info Gain", "Gain Ratio", "Gini Index"], state='readonly')
        self.classifierSplitTypeCB.current(0)
        self.classifierSplitTypeCB.pack(side=tkinter.LEFT, fill=tkinter.X, padx=5, pady=5)

        thirdFrame = tkinter.Frame(frame, bg="lightblue")
        thirdFrame.pack(side=tkinter.TOP, fill=tkinter.X, padx=5, pady=5)
        discretizationTypeLB = tkinter.Label(thirdFrame, text="Discretization type: ", height=1, width=23, bg="lightblue")
        discretizationTypeLB.pack(side=tkinter.LEFT, fill=tkinter.X, padx=5, pady=5)
        self.discretizationTypeCB = ttk.Combobox(thirdFrame, values=["Equal Depth", "Equal Width", "Entropy", "Gini Index"],
                                                 state='readonly')
        self.discretizationTypeCB.current(2)
        self.discretizationTypeCB.pack(side=tkinter.LEFT, fill=tkinter.X, padx=5, pady=5)

        fourthFrame = tkinter.Frame(frame, bg="lightblue")
        fourthFrame.pack(side=tkinter.TOP, fill=tkinter.X, padx=5, pady=5)
        discretizationBinsLB = tkinter.Label(fourthFrame, text="Discretization Bins: ", height=1, width=23, bg="lightblue")
        discretizationBinsLB.pack(side=tkinter.LEFT, fill=tkinter.X, padx=5, pady=5)
        self.discretizationBinsTF = tkinter.Text(fourthFrame, height=1, width=23, bg="White")
        self.discretizationBinsTF.pack(side=tkinter.LEFT, fill=tkinter.X, padx=5, pady=5)

        fifthFrame = tkinter.Frame(frame, bg="lightblue")
        fifthFrame.pack(side=tkinter.TOP, fill=tkinter.X, padx=5, pady=5)
        filePathLB = tkinter.Label(fifthFrame, text="Csv file path: ", height=1, width=23, bg="lightblue")
        filePathLB.pack(side=tkinter.LEFT, fill=tkinter.X, padx=5, pady=5)
        self.filePathTF = tkinter.Text(fifthFrame, height=1, width=23, bg="White")
        self.filePathTF.pack(side=tkinter.LEFT, fill=tkinter.X, padx=5, pady=5)
        filePathBT = tkinter.Button(fifthFrame, text="Browse", command=self.loadCsvFilePath)
        filePathBT.pack(side=tkinter.LEFT, fill=tkinter.X, padx=5, pady=5)

        sixFrame = tkinter.Frame(frame, bg="lightblue")
        sixFrame.pack(side=tkinter.TOP, fill=tkinter.X, padx=5, pady=5)
        savingFolderPathLB = tkinter.Label(sixFrame, text="Path for saving files: ", height=1, width=23, bg="lightblue")
        savingFolderPathLB.pack(side=tkinter.LEFT, fill=tkinter.X, padx=5, pady=5)
        self.savingfolderPathTF = tkinter.Text(sixFrame, height=1, width=23, bg="White")
        self.savingfolderPathTF.pack(side=tkinter.LEFT, fill=tkinter.X, padx=5, pady=5)
        savingfolderPathBT = tkinter.Button(sixFrame, text="Browse", command=self.loadSavingFolderPath)
        savingfolderPathBT.pack(side=tkinter.LEFT, fill=tkinter.X, padx=5, pady=5)

        startBuildBT = tkinter.Button(frame, text="Start Build", width=23, command=self.startBuild)
        startBuildBT.pack(side=tkinter.BOTTOM, padx=5, pady=5)

        self.currentFrame = frame

    def buildMessageFrame(self):
        """"
        method to build frame of process massages
        """
        self.currentFrame.destroy()

        frame = tkinter.Frame(self.root, bg="lightblue")
        frame.pack(side=tkinter.TOP, fill=tkinter.X, padx=5, pady=5)

        self.messageLB = tkinter.Label(frame, text="", bg="White")
        self.messageLB.pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=tkinter.YES, padx=5, pady=5)

        goBackBT = tkinter.Button(frame, text="Back", width=23, command=self.buildBuildingClassifierFrame)
        goBackBT.pack(side=tkinter.BOTTOM, padx=5, pady=5)

        self.currentFrame = frame

    def loadCsvFilePath(self):
        """"
        method to load chosen file path to text field
        """
        folderPath = filedialog.askopenfilename(initialdir="/", title="Select file", filetypes=(("Csv Files", ".csv"),))
        self.filePathTF.insert(tkinter.END, folderPath)

    def loadSavingFolderPath(self):
        """"
        method to load chosen folder path to save files in text field
        """
        folderPath = filedialog.askdirectory(initialdir="/", title="Select file")
        self.savingfolderPathTF.insert(tkinter.END, folderPath)

    def startBuild(self):
        """"
        method to start classifier building process
        """
        try:
            self.checkValues()
        except ValueError:
            return
        builder = BuildClassifierProcess()
        builder.setFolderPath(self.filePathTF.get("1.0", tkinter.END).rstrip()).setClassifierType(self.classifierTypeCB.get())\
            .setClassifierSplitType(self.classifierSplitTypeCB.get()).setDiscretizationType(self.discretizationTypeCB.get()).\
            setDiscretizationBins(int(self.discretizationBinsTF.get("1.0", tkinter.END)))\
            .setSavingFolderPath(self.savingfolderPathTF.get("1.0", tkinter.END).rstrip())
        self.buildMessageFrame()
        threading.Thread(target=builder.startProcess, args=(self.messageLB,)).start()

    def buildClassifierBT_event(self):
        """"
        method to move between main screen to building classifier screen
        """
        self.currentFrame.destroy()
        self.currentFrame = self.buildBuildingClassifierFrame()

    def lockComboBoxForNaiveByase(self, event):
        """"
        method lock split function type when naive bayes chosen
        """
        if self.classifierTypeCB.get().upper() == "ID3":
            self.classifierSplitTypeCB.config(state='readonly')
        else:
            self.classifierSplitTypeCB.config(state='disabled')

    def checkValues(self):
        """"
        method validate inputs
        Raises:
            ValueError: if input is invalid
        """
        try:
            value = int(self.discretizationBinsTF.get("1.0", tkinter.END))
            if value <= 1:
                raise ValueError
        except ValueError:
            messagebox.showinfo("Alert", "Discretization bins input invalid")
            raise ValueError
        if self.filePathTF.get("1.0", tkinter.END).rstrip() == "":
            messagebox.showinfo("Alert", "csv file path invalid")
            raise ValueError
        if self.savingfolderPathTF.get("1.0", tkinter.END).rstrip() == "":
            messagebox.showinfo("Alert", "Path for saving files invalid")
            raise ValueError


if __name__ == '__main__':
    Gui()
