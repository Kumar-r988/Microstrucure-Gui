import sys,subprocess
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QProgressBar, QStackedWidget, QFileDialog, QMessageBox
from PyQt5 import QtGui, QtWidgets, QtCore
from Microstructure import crop, polySim as pm
import os
from PyQt5.QtCore import QThread, pyqtSignal
from skimage import exposure, io
from micro1 import microgen as mi
import cv2
import numpy as np
import matplotlib.pyplot as plt
from interlamellar_spacing import random_interlamellar_spacing as mrs
from clean import noise_clean as nc

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Page 0")
        self.setGeometry(100, 100, 1350, 855)
        
        num_phases = 0
        folder_page2 = ""
        num_micrographs = 0
        nucleation_rate = 0
        growth_rate = 0
        micrograph_size = 0
        phase_names=''
        phase_fractions=''
        selected_directories=''
        image_path=None

        self.stacked_widget = QStackedWidget()
        self.page0 = QWidget()
        
        self.poly=PolyCrystal(self)
        #Artificial Microstructure Module
        # self.art_micro1 = ArtificialMicrostructureWindow1(self)
        self.art_micro2 = ArtificialMicrostructureWindow2(self)
        self.art_micro3 = ArtificialMicrostructureWindow3(self, num_phases, folder_page2, num_micrographs, nucleation_rate, growth_rate, micrograph_size)
        self.art_micro4 =ArtificialMicrostructureWindow4(self,num_phases,folder_page2,phase_names,phase_fractions,selected_directories,num_micrographs,nucleation_rate,growth_rate,micrograph_size)

        # Microstructure Cleaning Module
        self.micro_clean1=MicrostructureCleaning1(self)
        self.micro_clean2=MicrostructureCleaning2(self,image_path)
        
        self.pearl1=PearliteLamellaOrientation1(self)
        # self.pearl2=PearliteLamellaOrientation2(self,image_path)
    
        self.setup_page0()

        self.stacked_widget.addWidget(self.page0)
        self.stacked_widget.addWidget(self.poly)
        # self.stacked_widget.addWidget(self.art_micro1)
        self.stacked_widget.addWidget(self.art_micro2)
        self.stacked_widget.addWidget(self.art_micro3)
        self.stacked_widget.addWidget(self.art_micro4)
        
        self.stacked_widget.addWidget(self.micro_clean1)
        self.stacked_widget.addWidget(self.micro_clean2)
        
        self.stacked_widget.addWidget(self.pearl1)
        self.setCentralWidget(self.stacked_widget)
        
        
    def setup_page0(self):
        self.pushButton = QtWidgets.QPushButton("Polycrystalline Template Generation", self.page0)
        self.pushButton.setGeometry(QtCore.QRect(490, 60, 291, 101))

        self.pushButton_2 = QtWidgets.QPushButton("Artificial Microstructure", self.page0)
        self.pushButton_2.setGeometry(QtCore.QRect(490, 180, 291, 101))

        self.pushButton_3 = QtWidgets.QPushButton("Microstructure Cleaning", self.page0)
        self.pushButton_3.setGeometry(QtCore.QRect(490, 300, 291, 101))

        self.pushButton_4 = QtWidgets.QPushButton("Pearlite Lamella Orientation Map", self.page0)
        self.pushButton_4.setGeometry(QtCore.QRect(490, 420, 291, 101))

        self.pushButton_5 = QtWidgets.QPushButton("ML Model Training", self.page0)
        self.pushButton_5.setGeometry(QtCore.QRect(490, 540, 291, 101))

        self.pushButton.clicked.connect(self.open_polycrystalline_template)
        self.pushButton_2.clicked.connect(self.open_artificial_microstructure2)
        self.pushButton_3.clicked.connect(self.open_micro_cleaning1)
        self.pushButton_4.clicked.connect(self.open_pearl1)

    def open_polycrystalline_template(self):
        self.stacked_widget.setCurrentWidget(self.poly)
    # def open_artificial_microstructure1(self):
    #     self.stacked_widget.setCurrentWidget(self.art_micro1)
    def open_artificial_microstructure2(self):
        self.stacked_widget.setCurrentWidget(self.art_micro2)
    
    def open_artificial_microstructure3(self):
        self.stacked_widget.setCurrentWidget(self.art_micro3)
    
    def open_artificial_microstructure4(self):
        self.stacked_widget.setCurrentWidget(self.art_micro4)
        
    def open_micro_cleaning1(self):
        self.stacked_widget.setCurrentWidget(self.micro_clean1)
        
    def open_micro_cleaning2(self):
        self.stacked_widget.setCurrentWidget(self.micro_clean2)
    
    def open_micro_cleaning3(self, thresholded_image_path):
        self.micro_cleaning3 = MicroStructureCleaning3(self, thresholded_image_path)
        self.stacked_widget.addWidget(self.micro_cleaning3)
        self.stacked_widget.setCurrentWidget(self.micro_cleaning3)
        
    def open_pearl1(self):
        self.stacked_widget.setCurrentWidget(self.pearl1)
        
    def open_pearl2(self, image_path):
            image_path = image_path  # Set the image path
            pearl2 = PearliteLamellaOrientation2(self, image_path)
            self.stacked_widget.addWidget(pearl2)
            self.stacked_widget.setCurrentWidget(pearl2)
               

    def go_back_to_page0(self):
        self.stacked_widget.setCurrentWidget(self.page0)
        
class PolyCrystal(QWidget):
    
    def create_folders(self):
        directory = QFileDialog.getExistingDirectory(None, "Select Directory")
        if directory:
            self.folder_selected = True
            self.folder_paths = []
            folder_names = ["Template Save Directory"]
            for folder_name in folder_names:
                new_folder_path = os.path.join(directory, folder_name)
                if not os.path.exists(new_folder_path):
                    os.mkdir(new_folder_path)
                    self.folder_paths.append(new_folder_path)
                else:
                    QMessageBox.information(None, "Folder Exists", f"The folder already exists: {new_folder_path}")
                    self.folder_paths.append(new_folder_path)  # Append even if folder exists
            print("Folders created:")
            for path in self.folder_paths:
                print(path)

            # Set the text of line edits with folder paths
            self.linedit_new.setText(directory)

            # Change the color of the button to green
            self.pushButton_2.setStyleSheet("background-color: green")
            return self.folder_paths
        else:
            self.folder_selected = False
            return None

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.selected_folder=None
        
        self.setWindowTitle("Polycrystalline Template Generation")
        self.setGeometry(100, 100, 1238, 854)
        
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(410, 30, 550, 51))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label.setText("Polycrystalline Template Generation")
        
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(320, 150, 310, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_2.setText("Enter the number of phases")
        
        self.linedit_new = QtWidgets.QLineEdit(self.centralwidget)
        self.linedit_new.setGeometry(QtCore.QRect(980, 450, 200, 41))
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(780, 150, 191, 41))
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit.setText("2")
        
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(320, 210, 221, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_4.setText("No. of Micrographs")
        
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(320, 270, 221, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_3.setText("Nucleation Rate")
        
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(320, 330, 221, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.label_5.setText("Growth Rate")
        
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(320, 390, 221, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.label_6.setText("Size of Micrograph")
        
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_2.setGeometry(QtCore.QRect(780, 210, 191, 41))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.lineEdit_2.setText("10")
        
        self.lineEdit_3 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_3.setGeometry(QtCore.QRect(780, 270, 191, 41))
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.lineEdit_3.setText("10")
        
        self.lineEdit_4 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_4.setGeometry(QtCore.QRect(780, 330, 191, 41))
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.lineEdit_4.setText("8")
        
        self.lineEdit_5 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_5.setGeometry(QtCore.QRect(780, 390, 191, 41))
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.lineEdit_5.setText("512")
        
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(540, 590, 300, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setText("Generate Micrographs")
        self.pushButton.clicked.connect(self.generate_and_save_micrographs)
        
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(320, 450, 285, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.label_7.setText("Save Directory")
        
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(320, 510, 300, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(780, 450, 191, 41))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.setText("Select Directory")
        self.pushButton_2.clicked.connect(lambda: self.create_folders())
        
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(100,100,100,100))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.setText("Back")
        self.pushButton_3.clicked.connect(self.main_window.go_back_to_page0)
        
        main_window.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(main_window)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1238, 26))
        self.menubar.setObjectName("menubar")
        main_window.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(main_window)
        self.statusbar.setObjectName("statusbar")
        main_window.setStatusBar(self.statusbar)
        
        layout = QVBoxLayout(self)
        layout.addWidget(self.centralwidget)
        
    def microsimulator(self, nucleation_rate, growth_rate, micrograph_size):
        try:
            img = np.zeros((micrograph_size, micrograph_size), dtype=np.uint16)
            ng, avramiExponent = pm.generateStructure(img, nucleation_rate, growth_rate, Height=300, RandomizeGrayValues=True)
            array_buffer = img.tobytes()
            print('number of grains, avrami exponent :: ', ng, avramiExponent)
            # plt.imshow(img)
            # plt.show()
            return img, ng, avramiExponent
        except Exception as e:
            print(f"Error in microsimulator: {e}")
            QMessageBox.critical(None, "Error", f"An error occurred in microsimulator: {e}")
            return None, None, None

    def save_micrograph(self, img, filename):
        try:
            plt.imsave(filename, img)
            plt.close()  # Ensure the plot is closed
        except Exception as e:
            print(f"Error in save_micrograph: {e}")
            QMessageBox.critical(None, "Error", f"An error occurred while saving the micrograph: {e}")

    def generate_and_save_micrographs(self):
        try:
            num_phases = int(self.lineEdit.text())
            num_micrographs = int(self.lineEdit_2.text())
            nucleation_rate = int(self.lineEdit_3.text())
            growth_rate = float(self.lineEdit_4.text())
            micrograph_size = int(self.lineEdit_5.text())
            save_directory = self.folder_paths[0]
            if num_phases and num_micrographs and nucleation_rate and growth_rate and micrograph_size and save_directory:
                for i in range(num_micrographs):
                    img, ng, avrami_exponent = self.microsimulator(nucleation_rate, growth_rate, micrograph_size)
                    if img is not None:
                        filename = os.path.join(save_directory, f"micrograph_{i+1}.png")
                        self.save_micrograph(img, filename)
                    else:
                        print("Skipping saving due to error in microsimulator")
                QMessageBox.information(None, "Success", f"{num_micrographs} micrographs saved successfully.")
        except Exception as e:
            print(f"Error in generate_and_save_micrographs: {e}")
            QMessageBox.critical(None, "Error", f"An error occurred during micrograph generation: {e}")
        
# class ArtificialMicrostructureWindow1(QWidget):
#     def __init__(self, main_window):
#         super().__init__()
#         self.main_window = main_window

#         self.setWindowTitle("Artificial Microstructure")
#         self.setGeometry(100, 100, 1238, 854)

#         self.centralwidget = QtWidgets.QWidget(self)
#         self.centralwidget.setObjectName("centralwidget")

#         self.backbutton1 = QtWidgets.QPushButton(self.centralwidget)
#         self.backbutton1.setObjectName("backbutton1")
#         self.backbutton1.setGeometry(QtCore.QRect(100, 100, 100, 100))
#         self.backbutton1.setText("Back")

#         self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
#         self.pushButton_2.setGeometry(QtCore.QRect(460, 217, 251, 61))
#         self.pushButton_2.setObjectName("pushButton_2")
#         self.pushButton_2.setText("New Project")

#         self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
#         self.pushButton_3.setGeometry(QtCore.QRect(460, 500, 251, 61))
#         self.pushButton_3.setObjectName("pushButton_3")
#         self.pushButton_3.setText("Resume Existing")

#         layout = QVBoxLayout(self)
#         layout.addWidget(self.centralwidget)

#         self.pushButton_2.clicked.connect(self.open_art_micro2)
#         self.backbutton1.clicked.connect(self.main_window.go_back_to_page0)

#     def open_art_micro2(self):
#         self.page_2 = ArtificialMicrostructureWindow2(self.main_window)
#         self.main_window.stacked_widget.addWidget(self.page_2)
#         self.main_window.stacked_widget.setCurrentWidget(self.page_2)
        
        
class ArtificialMicrostructureWindow2(QWidget):
    def create_folders(self):
        directory = QFileDialog.getExistingDirectory(None, "Select Directory")
        if directory:
            self.folder_selected = True
            self.folder_paths = []
            folder_names = ["Micrograph Save Directory", "Ground Truth Save Directory"]
            for folder_name in folder_names:
                new_folder_path = os.path.join(directory, folder_name)
                if not os.path.exists(new_folder_path):
                    os.mkdir(new_folder_path)
                    self.folder_paths.append(new_folder_path)
                else:
                    QMessageBox.information(None, "Folder Exists", f"The folder already exists: {new_folder_path}")
                    self.folder_paths.append(new_folder_path)  # Append even if folder exists
            print("Folders created:")
            for path in self.folder_paths:
                print(path)

            # Set the text of line edits with folder paths
            self.lineEdit_6.setText(directory)
            
            #self.lineEdit_2.setText(self.folder_paths[1])

            # Change the color of the button to green
            self.pushButton_3.setStyleSheet("background-color: green")
            self.pushButton.setEnabled(True)
            return self.folder_paths
        else:
            self.folder_selected = False
            return None

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.selected_folder_2 = None  # Initialize variable for selected folder for pushButton_2
        self.selected_folder_3 = None  # Initialize variable for selected folder for pushButton_3
    

        self.setWindowTitle("Artificial Microstructure")
        self.setGeometry(100, 100, 1238, 854)

        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(470, 30, 301, 51))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label.setText("Artificial Microstructure")

        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(320, 150, 310, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")

        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(780, 150, 191, 41))
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit.setText("2")

        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(320, 210, 221, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")

        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(320, 270, 221, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")

        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(320, 330, 221, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")

        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(320, 390, 221, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")

        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_2.setGeometry(QtCore.QRect(780, 210, 191, 41))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.lineEdit_2.setText("10")

        self.lineEdit_3 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_3.setGeometry(QtCore.QRect(780, 270, 191, 41))
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.lineEdit_3.setText("10")

        self.lineEdit_4 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_4.setGeometry(QtCore.QRect(780, 330, 191, 41))
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.lineEdit_4.setText("8")

        self.lineEdit_5 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_5.setGeometry(QtCore.QRect(780, 390, 191, 41))
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.lineEdit_5.setText("255")

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(460, 620, 251, 61))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")

        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(780, 450, 191, 41))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.setText("Select Directory")
        self.pushButton_3.clicked.connect(self.create_folders)

        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.label_7.setGeometry(QtCore.QRect(320, 450, 285, 31))
        self.label_7.setText("Save Directory")

        self.lineEdit_6 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.lineEdit_6.setGeometry(QtCore.QRect(990, 450, 191, 41))

        self.label_2.setText("Enter The Number of Phases")
        self.label_4.setText("No. of Micrographs")
        self.label_3.setText("Nucleation Rate")
        self.label_5.setText("Growth Rate")
        self.label_6.setText("Size Of Micrograph")
        self.pushButton.setText("Next")
        self.pushButton.setEnabled(False)
        self.pushButton.clicked.connect(self.open_art_micro3)

        self.backbutton2 = QtWidgets.QPushButton(self.centralwidget)
        self.backbutton2.setGeometry(QtCore.QRect(100, 100, 100, 100))
        self.backbutton2.setObjectName("backbutton2")
        self.backbutton2.setText("Back")

        layout = QVBoxLayout(self)
        layout.addWidget(self.centralwidget)

        self.backbutton2.clicked.connect(self.main_window.go_back_to_page0)

    def get_user_inputs(self):
        user_inputs = {}
        self.selected_folder_2=self.folder_paths[0]
        self.selected_folder_3=self.folder_paths[1]

        user_inputs['folder_page2'] = [self.selected_folder_2, self.selected_folder_3]
        try:
            user_inputs['num_phases'] = int(self.lineEdit.text())
            user_inputs['num_micrographs'] = int(self.lineEdit_2.text())
            user_inputs['nucleation_rate'] = int(self.lineEdit_3.text())
            user_inputs['growth_rate'] = float(self.lineEdit_4.text())
            user_inputs['micrograph_size'] = int(self.lineEdit_5.text())
        except ValueError:
            QMessageBox.warning(None, "Warning", "Please enter valid integer and float values.")
            return None

        if user_inputs['nucleation_rate'] <= user_inputs['growth_rate']:
            QMessageBox.warning(None, "Warning", "Nucleation rate should be greater than growth rate.")
            return None
        print(user_inputs)
        return user_inputs

    def open_art_micro3(self):
        if not hasattr(self, 'folder_paths') or not self.folder_paths:
            folder_paths = self.create_folders()
            if not folder_paths:
                QMessageBox.warning(None, "Warning", "Please select a directory first.")
                return
        else:
            folder_paths = self.folder_paths
            
        user_inputs = self.get_user_inputs()
        if user_inputs:
            self.page3 = ArtificialMicrostructureWindow3(
                self.main_window,
                user_inputs['num_phases'],
                user_inputs['folder_page2'],
                user_inputs['num_micrographs'],
                user_inputs['nucleation_rate'],
                user_inputs['growth_rate'],
                user_inputs['micrograph_size']
            )
            self.main_window.stacked_widget.addWidget(self.page3)
            self.main_window.stacked_widget.setCurrentWidget(self.page3)          
            
            

class ArtificialMicrostructureWindow3(QWidget):
    def __init__(self, main_window, num_phases, folder_page2, num_micrographs, nucleation_rate, growth_rate, micrograph_size):
        super().__init__()

        self.main_window = main_window
        self.num_phases = num_phases
        self.folder_page2 = folder_page2
        print(folder_page2)
        self.num_micrographs = num_micrographs
        self.nucleation_rate = nucleation_rate
        self.growth_rate = growth_rate
        self.micrograph_size = micrograph_size

        self.phase_name_edits = []
        self.phase_fraction_edits = []
        self.labels = []

        self.setWindowTitle("Label the Phases")
        self.setGeometry(100, 100, 1238, 857)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.setup_ui()

    def setup_ui(self):
        font = QtGui.QFont()
        font.setPointSize(16)

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(470, 30, 350, 51))
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label.setText("Please Label the Phases")

        self.label_creator(self.num_phases)

        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(540, 760, 151, 41))
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.setText("Next")
        self.pushButton_2.clicked.connect(self.open_art_micro4)

        self.backbutton3 = QtWidgets.QPushButton(self.centralwidget)
        self.backbutton3.setGeometry(QtCore.QRect(100, 100, 151, 41))
        self.backbutton3.setFont(font)
        self.backbutton3.setObjectName("backbutton3")
        self.backbutton3.setText("Back")
        self.backbutton3.clicked.connect(self.main_window.open_artificial_microstructure2)

        layout = QVBoxLayout(self)
        layout.addWidget(self.centralwidget)

    def label_creator(self, n):
        layout = QVBoxLayout()

        for i in range(n):
            hbox = QHBoxLayout()

            label1 = QtWidgets.QLabel(f'Phase {i+1}:')
            self.labels.append(label1)

            line_edit1 = QtWidgets.QLineEdit()
            self.phase_name_edits.append(line_edit1)
            line_edit1.setText("Pearlite")

            label2 = QtWidgets.QLabel(f'Fraction:')
            self.labels.append(label2)

            line_edit2 = QtWidgets.QLineEdit()
            self.phase_fraction_edits.append(line_edit2)
            line_edit2.setText("0.5")

            hbox.addWidget(label1)
            hbox.addWidget(line_edit1)
            hbox.addWidget(label2)
            hbox.addWidget(line_edit2)

            layout.addLayout(hbox)

        self.centralwidget.setLayout(layout)

        for i in range(n):
            self.labels[i*2].setGeometry(200, 100 + i * 150, 100, 30)
            self.phase_name_edits[i].setGeometry(320, 100 + i * 150, 100, 30)
            self.labels[i*2+1].setGeometry(440, 100 + i * 150, 100, 30)
            self.phase_fraction_edits[i].setGeometry(560, 100 + i * 150, 100, 30)

    def get_phase_labels(self):
        phase_labels = []
        phase_fractions=[]
        selected_directories=[]
        total_fraction = 0.0
        

        for i in range(self.num_phases):
            phase_name = self.phase_name_edits[i].text()
            phase_fraction = float(self.phase_fraction_edits[i].text())

            if phase_fraction < 0 or phase_fraction > 1:
                QMessageBox.warning(None, "Warning", "Phase fraction should be between 0 and 1.")
                return None

            total_fraction += phase_fraction
            phase_labels.append(phase_name)
            phase_fractions.append(phase_fraction)
            
            folder_path = os.path.join(self.folder_page2[0], phase_name)
            if os.path.exists(folder_path):
                # Folder exists, copy its path
                 selected_directories.append(folder_path)
            else:
                # Folder doesn't exist, create it
                os.makedirs(folder_path)
                selected_directories.append(folder_path)
                

        if total_fraction != 1:
            QMessageBox.warning(None, "Warning", "Sum of phase fractions should be equal to 1.")
            return None
        print(phase_labels,phase_fractions,selected_directories)
        return phase_labels,phase_fractions,selected_directories
        
    
    def user_inputs_page_3(self):
        # Create a dictionary to store the inputs
        phase_names,phase_fractions,selected_directories=self.get_phase_labels()
        page_3_inputs = {}

        # Store the number of phases
        page_3_inputs["num_phases"] = self.num_phases
        page_3_inputs['folder_page2'] = self.folder_page2
        page_3_inputs['phase_names'] = phase_names
        page_3_inputs['phase_fractions'] = phase_fractions
        page_3_inputs['selected_directories']=selected_directories
        page_3_inputs['num_micrographs'] = self.num_micrographs
        page_3_inputs['nucleation_rate'] =self.nucleation_rate
        page_3_inputs['growth_rate'] = self.growth_rate
        page_3_inputs['micrograph_size'] =self.micrograph_size

        # print(page_3_inputs)  # Debug statement
        return page_3_inputs
    
    def open_art_micro4(self):
        user_inputs = self.user_inputs_page_3()
        print(user_inputs)
        if user_inputs:
            self.page4= ArtificialMicrostructureWindow4(
                self.main_window,
                user_inputs['num_phases'],
                user_inputs['folder_page2'],
                user_inputs['phase_names'],
                user_inputs['phase_fractions'],
                user_inputs['selected_directories'],
                user_inputs['num_micrographs'],
                user_inputs['nucleation_rate'],
                user_inputs['growth_rate'],
                user_inputs['micrograph_size']
            )
            self.main_window.stacked_widget.addWidget(self.page4)
            self.main_window.stacked_widget.setCurrentWidget(self.page4)
    






class WorkerThread(QThread):
    updateProgress = pyqtSignal(int)

    def __init__(self, selected_directories, phase_fractions, micrograph_size, nucleation_rate, growth_rate, folders_page2, r):
        super().__init__()
        self.selected_directories = [path + "/" if not path.endswith("/") else path for path in selected_directories]
        self.phase_fractions = phase_fractions
        self.micrograph_size = micrograph_size
        self.nucleation_rate = nucleation_rate
        self.growth_rate = growth_rate
        self.folders_page2 = [path + "/" if not path.endswith("/") else path for path in folders_page2]
        self.r = r

    def run(self):
        for i in range(1, self.r + 1):
            y = int(100 / self.r)
            img, ground = mi.microstructure(
                self.selected_directories, self.phase_fractions, 
                self.micrograph_size, self.nucleation_rate, self.growth_rate
            )
            img_adjusted = exposure.rescale_intensity(img, in_range='image', out_range='dtype')
            img_save_path = f"{self.folders_page2[0]}/img{i}.png"  # Adjust the save path for images
            ground_save_path = f"{self.folders_page2[1]}/gt{i}.png"
            io.imsave(img_save_path, img_adjusted)
            io.imsave(ground_save_path, ground)
            self.updateProgress.emit(y * i)
    
    
class ArtificialMicrostructureWindow4(QWidget):
    def __init__(self,main_window,num_phases,folder_page2,phase_names,phase_fractions,selected_directories,num_micrographs,nucleation_rate,growth_rate,micrograph_size):
        super().__init__()
        self.main_window = main_window
        self.num_phases = num_phases
        self.folder_page2 = folder_page2
        self.phase_names=phase_names
        self.phase_fractions=phase_fractions
        self.selected_directories=selected_directories
        self.num_micrographs = num_micrographs
        self.nucleation_rate = nucleation_rate
        self.growth_rate = growth_rate
        self.micrograph_size = micrograph_size
        
        
        self.buttons=[]
        
        self.setWindowTitle("Crop the Microstructures")
        self.setGeometry(100, 100, 1250, 857)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.setup_ui()
  
    def setup_ui(self):
        font = QtGui.QFont()
        font.setPointSize(16)
        
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(470, 30, 800, 51))
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label.setText("Please select Microstructure to crop the images")
        
        # Use phase_names and selected_directories provided as command-line arguments
        self.button_creator(self.phase_names, self.selected_directories)
        
        self.progress = QProgressBar(self.centralwidget)
        self.progress.setGeometry(300, 760, 600, 30)

        self.startButton = QPushButton(self.centralwidget)
        self.startButton.setGeometry(540, 700, 250, 41)
        self.startButton.setFont(font)
        self.startButton.setText("Start Generation")
        self.startButton.clicked.connect(self.start_microstructure_generation)
        # self.pushButton.setEnabled(False)
        # self.pushButton.clicked.connect(self.open_art_micro5)
        
        self.backbutton2 = QtWidgets.QPushButton(self.centralwidget)
        self.backbutton2.setGeometry(QtCore.QRect(100, 100, 151, 41))
        self.backbutton2.setFont(font)
        self.backbutton2.setObjectName("backbutton4")
        self.backbutton2.setText("Back")
        self.backbutton2.clicked.connect(self.main_window.open_artificial_microstructure2)

        layout = QVBoxLayout(self)
        layout.addWidget(self.centralwidget)

   
    def button_creator(self, phase_names, selected_directories):
        layout = QVBoxLayout()
        for phase_name, directory in zip(phase_names, selected_directories):
            hbox = QtWidgets.QHBoxLayout()

            button = QPushButton(f'Crop microstructure for {phase_name}')
            button.clicked.connect(lambda _, dir=directory: crop.crop_images(dir))  # Connect to crop_images function
            
            hbox.addWidget(button)

            layout.addLayout(hbox)

            # Append button to the list
            self.buttons.append(button)

        self.centralwidget.setLayout(layout)

    def start_microstructure_generation(self):
        self.workerThread = WorkerThread(
            self.selected_directories, self.phase_fractions, 
            self.micrograph_size, self.nucleation_rate, 
            self.growth_rate, self.folder_page2, self.num_micrographs
        )
        self.workerThread.updateProgress.connect(self.updateProgressBar)
        self.startButton.setEnabled(False)
        self.workerThread.start()

    def updateProgressBar(self, value):
        self.progress.setValue(value)
        if value == 100:
            self.startButton.setEnabled(True)




#Microcleaning Module
class CropWidget(QtWidgets.QLabel):
    def __init__(self, main_window, parent=None):
        super(CropWidget, self).__init__(parent)
        self.main_window = main_window
        self.start_x = self.start_y = self.end_x = self.end_y = None
        self.cropping = False

    def crop(self):
        self.start_x = self.start_y = self.end_x = self.end_y = None
        self.update()
        main_window.pushButton_2.setText("Crop Now")
        main_window.pushButton_2.setStyleSheet("background-color: green; color: white;")



    def mousePressEvent(self, event):
        if self.cropping:
            self.start_x = event.pos().x()
            self.start_y = event.pos().y()
            print(f"Mouse pressed at: {self.start_x}, {self.start_y}")

    def mouseMoveEvent(self, event):
        if self.cropping:
            self.end_x = event.pos().x()
            self.end_y = event.pos().y()
            self.update()
            print(f"Mouse moved to: {self.end_x}, {self.end_y}")

    def mouseReleaseEvent(self, event):
        if self.cropping:
            self.cropping = False
            self.end_x = event.pos().x()
            self.end_y = event.pos().y()
            self.save_cropped_image()
            print(f"Mouse released at: {self.end_x}, {self.end_y}")

    def paintEvent(self, event):
        super(CropWidget, self).paintEvent(event)
        if self.cropping and self.start_x is not None and self.end_x is not None:
            qp = QtGui.QPainter(self)
            qp.setPen(QtGui.QPen(QtCore.Qt.green, 2))
            rect = QtCore.QRect(QtCore.QPoint(self.start_x, self.start_y), QtCore.QPoint(self.end_x, self.end_y))
            qp.drawRect(rect)
            print(f"Rectangle drawn from: {self.start_x}, {self.start_y} to {self.end_x}, {self.end_y}")

    def enable_cropping(self):
        self.cropping = True
        print("Cropping enabled")
        main_window.pushButton_2.setText("Crop Now")
        main_window.pushButton_2.setStyleSheet("background-color: green; color: white;")

    def save_cropped_image(self):
        if self.start_x is not None and self.start_y is not None and self.end_x is not None and self.end_y is not None:
            pixmap = self.pixmap()
            if pixmap.isNull():
                print("Pixmap is null")
                return
            img = pixmap.toImage()
            scale_factor_x = img.width() / self.width()
            scale_factor_y = img.height() / self.height()
            start_x = int(self.start_x * scale_factor_x)
            start_y = int(self.start_y * scale_factor_y)
            end_x = int(self.end_x * scale_factor_x)
            end_y = int(self.end_y * scale_factor_y)
            cropped_img = img.copy(start_x, start_y, end_x - start_x, end_y - start_y)
            cropped_pixmap = QtGui.QPixmap.fromImage(cropped_img)
            self.save_path = "cropped_image.jpg"
            cropped_pixmap.save(self.save_path)
            self.show_confirmation_window()
            print(f"Cropped image saved at: {self.save_path}")

    def show_confirmation_window(self):
        confirmation_window = ConfirmationWindow(self.save_path, self.main_window)
        confirmation_window.exec_()
        if confirmation_window.proceed:
            self.open_micro_clean2()

    def open_micro_clean2(self):
        self.page2=MicrostructureCleaning2(self.main_window,self.save_path)
        main_window.stacked_widget.addWidget(self.page2)
        main_window.stacked_widget.setCurrentWidget(self.page2)

class ConfirmationWindow(QtWidgets.QDialog):
    def __init__(self, image_path, main_window, parent=None):
        super(ConfirmationWindow, self).__init__(parent)
        self.setWindowTitle("Crop Confirmation")
        self.image_path = image_path
        self.proceed = False
        self.main_window = main_window

        image_label = QtWidgets.QLabel()
        pixmap = QtGui.QPixmap(image_path)
        image_label.setPixmap(pixmap)
        image_label.setAlignment(QtCore.Qt.AlignCenter)

        proceed_button = QtWidgets.QPushButton("Proceed to Next Page")
        proceed_button.clicked.connect(self.proceed_to_next_page)

        crop_again_button = QtWidgets.QPushButton("Crop Again")
        crop_again_button.clicked.connect(self.crop_again)

        button_layout = QHBoxLayout()
        button_layout.addWidget(proceed_button)
        button_layout.addWidget(crop_again_button)

        layout = QVBoxLayout(self)
        layout.addWidget(image_label)
        layout.addLayout(button_layout)

    def proceed_to_next_page(self):
        self.proceed = True
        self.accept()
        # main_window.open_micro_cleaning2(self.image_path)

    def crop_again(self):
        self.proceed = False
        self.accept()
        main_window.pushButton_2.setText("Crop Now")
        main_window.pushButton_2.setStyleSheet("background-color: green; color: white;")
        main_window.pushButton_2.setEnabled(True)


class MicrostructureCleaning1(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setWindowTitle("Microstructure Cleaning")
        self.setGeometry(100, 100, 1238, 854)

        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(420, 80, 325, 61))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label.setText("Microstructure Cleaning")

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(470, 200, 250, 41))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setText("Upload Single Phase Micrograph")

        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(300, 660, 121, 41))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.setText("Crop this image")

        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(740, 660, 141, 41))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.setText("Take Full Image")

        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(300, 270, 581, 351))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")

        main_window.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(main_window)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1229, 26))
        self.menubar.setObjectName("menubar")
        main_window.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(main_window)
        self.statusbar.setObjectName("statusbar")
        main_window.setStatusBar(self.statusbar)

        self.file_path = None

        self.pushButton.clicked.connect(self.open_image)
        self.pushButton_3.clicked.connect(self.open_full_image)
        self.pushButton_2.clicked.connect(self.enable_crop_mode)

        self.pushButton_2.setEnabled(False)
        self.pushButton_3.setEnabled(False)

        self.Backbutton = QtWidgets.QPushButton(self.centralwidget)
        self.Backbutton.setGeometry(QtCore.QRect(100, 100, 100, 100))
        self.Backbutton.setObjectName("Backbutton")
        self.Backbutton.setText("Back")
        self.Backbutton.clicked.connect(self.main_window.go_back_to_page0)

        self.original_button_text = self.pushButton_2.text()
        self.original_button_style = self.pushButton_2.styleSheet()

        layout = QVBoxLayout(self)
        layout.addWidget(self.centralwidget)

        self.crop_widget = None  # New instance variable

    def open_image(self):
        self.file_path, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)")
        if self.file_path:
            pixmap = QtGui.QPixmap(self.file_path)
            if not pixmap.isNull():
                label_width = min(self.frame.width(), pixmap.width())
                label_height = min(self.frame.height(), pixmap.height())
                scaled_pixmap = pixmap.scaled(label_width, label_height, QtCore.Qt.KeepAspectRatio)
                label_x = (self.frame.width() - label_width) // 2
                label_y = (self.frame.height() - label_height) // 2

                if self.crop_widget:  # Delete previous crop widget if it exists
                    self.crop_widget.deleteLater()

                self.crop_widget = CropWidget(self.main_window, self.frame)  # Create a new crop widget
                self.crop_widget.setGeometry(label_x, label_y, label_width, label_height)
                self.crop_widget.setPixmap(scaled_pixmap)
                self.crop_widget.setAlignment(QtCore.Qt.AlignCenter)
                self.crop_widget.show()

                self.pushButton_2.setEnabled(True)
                self.pushButton_3.setEnabled(True)
                print("Image loaded and displayed")

    def enable_crop_mode(self):
        if self.crop_widget:
            self.crop_widget.enable_cropping()

    def show_original_button(self):
        self.pushButton_2.setText(self.original_button_text)
        self.pushButton_2.setStyleSheet(self.original_button_style)
        print("Reset button to original state")

    def open_micro_cleaning2(self):
        if self.file_path:
            self.page2 = MicrostructureCleaning2(self.main_window, self.file_path)
            self.main_window.stacked_widget.addWidget(self.page2)
            self.main_window.stacked_widget.setCurrentWidget(self.page2)

    def open_full_image(self):
        self.open_micro_cleaning2()


class MicrostructureCleaning2(QWidget):
    def __init__(self, main_window, file_path):
        super().__init__()
        self.main_window = main_window
        self.image_path = file_path
        print(self.image_path)
        self.setWindowTitle("Microstructure Cleaning")
        self.setGeometry(100, 100, 1238, 854)

        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(100, 80, 401, 281))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.frame.setStyleSheet("background-color: rgba(0, 0, 0, 0.7); color: white;")
        
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setGeometry(QtCore.QRect(740, 80, 401, 281))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.frame_2.setStyleSheet("background-color: rgba(0, 0, 0, 0.7); color: white;")
        
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(240, 400, 120, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label.setText("Actual Image")
        
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(880, 400, 150, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_2.setText("Threshold Image")
        
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(250, 510, 171, 51))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setText("Manual Thresholding")
        
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(250, 575, 171, 51))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.setText("Use Otsu Thresholding")
        self.pushButton_2.clicked.connect(self.otsu_thresholding)
        
        self.horizontalSlider = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider.setGeometry(QtCore.QRect(480, 520, 351, 22))
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider") 
        self.horizontalSlider.setMinimum(0)
        self.horizontalSlider.setMaximum(255)
        
        self.label_slider_value = QtWidgets.QLabel(self.centralwidget)
        self.label_slider_value.setGeometry(QtCore.QRect(480, 535, 351, 31))
        self.label_slider_value.setAlignment(QtCore.Qt.AlignCenter)
        self.label_slider_value.setObjectName("label_slider_value")
        self.label_slider_value.setText("0")  # Initial value
        
        self.horizontalSlider.valueChanged.connect(self.update_slider_value_label)
        
        self.label_x_thr = QtWidgets.QLabel(self.centralwidget)
        self.label_x_thr.setGeometry(QtCore.QRect(260, 640, 171, 31))
        self.label_x_thr.setObjectName("label_x_thr")
        self.label_x_thr.setText("Threshold Value: ")

        self.label_x_thr_value = QtWidgets.QLabel(self.centralwidget)
        self.label_x_thr_value.setGeometry(QtCore.QRect(650, 640, 71, 31))
        self.label_x_thr_value.setObjectName("label_x_thr_value")
        self.label_x_thr_value.setText("0")
        
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(590, 690, 131, 51))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_4.setText("Next")
        self.pushButton_4.clicked.connect(self.open_next)
        
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(390, 690, 131, 51))
        self.pushButton_5.setText("Back")
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_5.clicked.connect(self.open_previous)
        
        self.label_image = QtWidgets.QLabel(self.centralwidget)
        self.label_image.setGeometry(QtCore.QRect(110, 90, 381, 261))
        self.label_image.setObjectName("label_image")
        self.label_image.setScaledContents(True)  # Scale the image to fit the label

        self.label_thresholded_image = QtWidgets.QLabel(self.centralwidget)
        self.label_thresholded_image.setGeometry(QtCore.QRect(750, 90, 381, 261))
        self.label_thresholded_image.setObjectName("label_thresholded_image")
        self.label_thresholded_image.setScaledContents(True)  # Scale the image to fit the label

        # Load and display the original image
        self.load_and_display_images()

        layout = QVBoxLayout(self)
        layout.addWidget(self.centralwidget)

    def load_and_display_images(self):
        image = cv2.imread(self.image_path)
        if image is None:
            #QtWidgets.QMessageBox.critical(self, "Error", "Failed to load image. Please check the file path.")
            return
        
        # Display the original image
        height, width, _ = image.shape
        qimage = QtGui.QImage(image.data, width, height, 3 * width, QtGui.QImage.Format_BGR888)
        pixmap = QtGui.QPixmap.fromImage(qimage)
        self.label_image.setPixmap(pixmap)
        
        # Display the thresholded image (initially empty)
        self.label_thresholded_image.clear()
        
    def update_slider_value_label(self, value):
        self.label_slider_value.setText(str(value))
        self.manual_thresholding()  # Call manual_thresholding whenever the slider value changes
        
    def otsu_thresholding(self):
        image = cv2.imread(self.image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        x_thr, thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.label_x_thr_value.setText(str(x_thr))
        pixmap = QtGui.QPixmap.fromImage(QtGui.QImage(thresholded_image.data, thresholded_image.shape[1], thresholded_image.shape[0], thresholded_image.strides[0], QtGui.QImage.Format_Grayscale8))
        self.label_thresholded_image.setPixmap(pixmap)
        self.horizontalSlider.setValue(int(x_thr))  # Set slider value to x_thr
        self.thresholded_image_path = self.save_temp_image(thresholded_image)

    def manual_thresholding(self):
        image = cv2.imread(self.image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        threshold_value = self.horizontalSlider.value()
        _, thresholded_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
        pixmap = QtGui.QPixmap.fromImage(QtGui.QImage(thresholded_image.data, thresholded_image.shape[1], thresholded_image.shape[0], thresholded_image.strides[0], QtGui.QImage.Format_Grayscale8))
        self.label_thresholded_image.setPixmap(pixmap)
        self.thresholded_image_path = self.save_temp_image(thresholded_image)
        
    def save_temp_image(self, image):
        temp_path = 'temp_thresholded_image.png'
        cv2.imwrite(temp_path, image)
        return temp_path
    
    def open_previous(self):
        self.main_window.open_micro_cleaning1()  # Method to open the first page
        self.close()  # Close the current page

    def open_next(self):
        main_window.open_micro_cleaning3(self.thresholded_image_path)
    
class DrawingWidget(QtWidgets.QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = QtGui.QImage(491, 331, QtGui.QImage.Format_Grayscale8)
        self.image.fill(0)  # Fill with black (0) initially
        self.last_point = QtCore.QPoint()
        self.pen_color = QtGui.QColor(255,255,255)  # Default white color for drawing
        self.pen_size = 5
        self.drawing = False
        self.mode = 'draw'  # Mode can be 'draw' or 'erase'
        self.setScaledContents(True)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(self.rect(), self.image)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.last_point = event.pos()
            self.drawing = True

    def mouseMoveEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton and self.drawing:
            painter = QtGui.QPainter(self.image)
            scale_x = self.image.width() / self.width()
            scale_y = self.image.height() / self.height()
            start_point = QtCore.QPointF(self.last_point.x() * scale_x, self.last_point.y() * scale_y)
            end_point = QtCore.QPointF(event.pos().x() * scale_x, event.pos().y() * scale_y)
            if self.mode == 'draw':
                painter.setPen(QtGui.QPen(self.pen_color, self.pen_size, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
            elif self.mode == 'erase':
                painter.setPen(QtGui.QPen(QtGui.QColor(0), self.pen_size, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
            painter.drawLine(start_point, end_point)
            self.last_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.drawing = False

    def load_image(self, image):
        height, width = image.shape
        bytes_per_line = width
        self.image = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_Grayscale8)
        self.update()

    def set_mode(self, mode):
        if mode in ['draw', 'erase']:
            self.mode = mode

    def set_pen_size(self, size):
        self.pen_size = size

    def save_image(self, file_path):
        self.image.save(file_path)


class MicroStructureCleaning3(QtWidgets.QWidget):
    def __init__(self, main_window=None, thresholded_image_path=None):
        super(MicroStructureCleaning3, self).__init__(parent=main_window)
        self.main_window = main_window
        self.thresholded_image_path = thresholded_image_path
        self.setupUi()
        self.load_initial_image(thresholded_image_path)
        
        
    def setupUi(self):
        self.setObjectName("MainWindow")
        self.resize(1240, 854)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")

        # Frame for displaying the default loaded image at the left bottom
        self.frame_default = QtWidgets.QFrame(self.centralwidget)
        self.frame_default.setGeometry(QtCore.QRect(100, 480, 491, 331))
        self.frame_default.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_default.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_default.setObjectName("frame_default")
        self.frame_default.setStyleSheet("border: 2px solid black;")
        
        self.label_default = QtWidgets.QLabel(self.frame_default)
        self.label_default.setGeometry(QtCore.QRect(0, 0, 491, 331))
        self.label_default.setObjectName("label_default")
        self.label_default.setScaledContents(True)

        # Frame for displaying the cleaned image at the top left
        self.frame_cleaned = QtWidgets.QFrame(self.centralwidget)
        self.frame_cleaned.setGeometry(QtCore.QRect(100, 80, 491, 331))
        self.frame_cleaned.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_cleaned.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_cleaned.setObjectName("frame_cleaned")
        self.frame_cleaned.setStyleSheet("border: 2px solid black;")
        
        self.label_cleaned = QtWidgets.QLabel(self.frame_cleaned)
        self.label_cleaned.setGeometry(QtCore.QRect(0, 0, 491, 331))
        self.label_cleaned.setObjectName("label_cleaned")
        self.label_cleaned.setScaledContents(True)

        # Frame for displaying the reconstructed image at the top right
        self.frame_reconstructed = QtWidgets.QFrame(self.centralwidget)
        self.frame_reconstructed.setGeometry(QtCore.QRect(680, 80, 491, 331))
        self.frame_reconstructed.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_reconstructed.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_reconstructed.setObjectName("frame_reconstructed")
        self.frame_reconstructed.setStyleSheet("border: 2px solid black;")

        self.label_reconstructed = DrawingWidget(self.frame_reconstructed)
        self.label_reconstructed.setGeometry(QtCore.QRect(0, 0, 491, 331))
        self.label_reconstructed.setObjectName("label_reconstructed")

        # Pen size slider
        self.slider_pen_size = QtWidgets.QSlider(QtCore.Qt.Horizontal, self.centralwidget)
        self.slider_pen_size.setGeometry(QtCore.QRect(821, 535, 200, 30))
        self.slider_pen_size.setRange(1, 50)
        self.slider_pen_size.setValue(5)
        self.slider_pen_size.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_pen_size.setTickInterval(5)
        self.slider_pen_size.valueChanged.connect(self.update_pen_size)

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(310, 430, 93, 28))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(870, 430, 93, 28))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(870, 500, 93, 28))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(870, 570, 93, 28))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(860, 700, 120, 40))
        self.pushButton_5.setObjectName("pushButton_5")

        self.pushButton_back = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_back.setGeometry(QtCore.QRect(10, 10, 93, 28))  # Adjust size and position as needed
        self.pushButton_back.setObjectName("pushButton_back")
        self.pushButton_back.setText("Back")
        self.pushButton_back.clicked.connect(self.go_back)
        # Set layout for the central widget
        self.setLayout(QtWidgets.QVBoxLayout())
        self.layout().addWidget(self.centralwidget)

        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

        self.pushButton.clicked.connect(self.cleaning)
        self.pushButton_2.clicked.connect(self.reconstruct)
        self.pushButton_3.clicked.connect(self.activate_drawing)
        self.pushButton_4.clicked.connect(self.activate_erasing)
        self.pushButton_5.clicked.connect(self.save_image)

        self.cleaned_image = None
        self.reconstructed_image = None
        
        
    def go_back(self):
        if self.main_window:
            main_window.open_micro_cleaning2()

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Clean"))
        self.pushButton_2.setText(_translate("MainWindow", "Reconstruct"))
        self.pushButton_3.setText(_translate("MainWindow", "Draw Manually"))
        self.pushButton_4.setText(_translate("MainWindow", "Erase"))
        self.pushButton_5.setText(_translate("MainWindow", "Save"))

    def display_image(self, label, image):
        height, width = image.shape
        bytes_per_line = width
        q_img = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap.fromImage(q_img).scaled(label.size(), QtCore.Qt.KeepAspectRatio)
        label.setPixmap(pixmap)

    def cleaning(self):
        global image
        if image is None:
            print("No image to clean.")
            return

        print("Cleaning image...")
        cleaned_img = nc.noise_clean(image)
        print("Cleaning done.")

        if cleaned_img is not None:
            self.display_image(self.label_cleaned, cleaned_img)
            self.cleaned_image = cleaned_img
            print("Cleaned image displayed successfully.")
        else:
            print("Cleaning failed. No image to display.")

    def reconstruct(self):
        if self.cleaned_image is not None:
            self.label_reconstructed.load_image(self.cleaned_image)
            self.reconstructed_image = self.cleaned_image
            self.pushButton_3.setEnabled(True)  # Enable drawing button
            self.pushButton_4.setEnabled(True)  # Enable erasing button
            print("Reconstructed image displayed successfully.")
        else:
            print("No cleaned image available to reconstruct.")
            
        

    def load_initial_image(self, image_path):
        print(f"Loading initial image from: {image_path}")
        global image
        image = cv2.imread(image_path, 0)
        if image is None:
            print("Failed to load initial image.")
            return
        
        self.display_image(self.label_default, image)
        print("Initial image loaded successfully.")

    def activate_drawing(self):
        if self.reconstructed_image is not None:
            self.label_reconstructed.set_mode('draw')
            self.pushButton_2.setText("Reset")

    def activate_erasing(self):
        if self.reconstructed_image is not None:
            self.label_reconstructed.set_mode('erase')
            self.pushButton_2.setText("Reset")

    def update_pen_size(self):
        size = self.slider_pen_size.value()
        self.label_reconstructed.set_pen_size(size)

    def save_image(self):
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(None, "Save Image", "", "PNG Files (*.png);;All Files (*)")
        if file_path:
            self.label_reconstructed.save_image(file_path)
            print(f"Image saved to: {file_path}")
         
class PearliteLamellaOrientation1(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        
        self.setWindowTitle("Pearlite Lamella Orientation")
        self.setGeometry(100, 100, 1238, 854)

        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(420, 80, 325, 61))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label.setText("Pearlite Lamella Orientation")
        
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(100, 210, 531, 361))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        
        self.imageLabel = QtWidgets.QLabel(self.frame)
        self.imageLabel.setGeometry(self.frame.rect())
        self.imageLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.imageLabel.setObjectName("imageLabel")
        
         # Set border style and default text
        self.imageLabel.setStyleSheet("border: 2px solid black;")
        self.imageLabel.setText("Upload Image Here")
        
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(800, 280, 200, 28))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setText('Upload Pearlite Micrograph')
        self.pushButton.clicked.connect(self.open_image)
        
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(450, 710, 150, 28))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.setText('Next')
        self.pushButton_3.setEnabled(False)
        self.pushButton_3.clicked.connect(self.open_next)
        
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(100,100,100,100))
        self.pushButton_4.setObjectName("pushButton_3")
        self.pushButton_4.setText("Back")
        self.pushButton_4.clicked.connect(self.main_window.go_back_to_page0)
        
        main_window.setCentralWidget(self.centralwidget)
        
        self.menubar = QtWidgets.QMenuBar(main_window)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1229, 26))
        self.menubar.setObjectName("menubar")
        main_window.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(main_window)
        self.statusbar.setObjectName("statusbar")
        main_window.setStatusBar(self.statusbar)
        
        self.file_path=None
        
        layout = QVBoxLayout(self)
        layout.addWidget(self.centralwidget)
        
    def open_image(self):
        self.file_path, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tif)")
        if self.file_path:
            pixmap = QtGui.QPixmap(self.file_path)
            if not pixmap.isNull():
                # Scale the pixmap to fit within the frame's dimensions while maintaining the aspect ratio
                scaled_pixmap = pixmap.scaled(self.frame.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
                self.imageLabel.setPixmap(scaled_pixmap)
                self.imageLabel.setText("")  # Clear the text when an image is loaded
                self.imageLabel.show()
        self.pushButton_3.setEnabled(True)
    
    def open_next(self):
        main_window.open_pearl2(self.file_path)  # Pass the thresholded image path to the next page
        self.close()


class CropWidget(QtWidgets.QLabel):
    def __init__(self,main_window, parent=None):
        super(CropWidget, self).__init__(parent)
        self.main_window = main_window
        self.start_x = self.start_y = self.end_x = self.end_y = None
        self.cropping = False

    def mousePressEvent(self, event):
        if self.cropping:
            self.start_x = event.pos().x()
            self.start_y = event.pos().y()

    def mouseMoveEvent(self, event):
        if self.cropping:
            self.end_x = event.pos().x()
            self.end_y = event.pos().y()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.cropping:
            self.cropping = False
            self.end_x = event.pos().x()
            self.end_y = event.pos().y()
            self.save_cropped_image()

    def paintEvent(self, event):
        super(CropWidget, self).paintEvent(event)
        if self.cropping and self.start_x is not None and self.end_x is not None:
            qp = QtGui.QPainter(self)
            qp.setPen(QtGui.QPen(QtCore.Qt.green, 2))
            rect = QtCore.QRect(QtCore.QPoint(self.start_x, self.start_y), QtCore.QPoint(self.end_x, self.end_y))
            qp.drawRect(rect)

    def enable_cropping(self):
        self.cropping = True

    def save_cropped_image(self):
        if self.start_x is not None and self.start_y is not None and self.end_x is not None and self.end_y is not None:
            # Get the pixmap from the label
            pixmap = self.pixmap()
            if pixmap.isNull():
                return
            # Get the original image
            img = pixmap.toImage()
            # Calculate the scaling factor
            scale_factor_x = img.width() / self.width()
            scale_factor_y = img.height() / self.height()
            # Calculate the cropping coordinates
            start_x = int(self.start_x * scale_factor_x)
            start_y = int(self.start_y * scale_factor_y)
            end_x = int(self.end_x * scale_factor_x)
            end_y = int(self.end_y * scale_factor_y)
            # Crop the image
            cropped_img = img.copy(start_x, start_y, end_x - start_x, end_y - start_y)
            # Convert the cropped image back to QPixmap
            cropped_pixmap = QtGui.QPixmap.fromImage(cropped_img)
            cropped_image_path = "cropped_image.png"
            cropped_pixmap.save(cropped_image_path)
            self.cropped_image_path = cropped_image_path
            self.cropped_image = cropped_img
            self.show_confirmation_window(cropped_image_path)
            
    def show_confirmation_window(self,cropped_image_path):
        confirmation_window = ConfirmationWindow(cropped_image_path)
        confirmation_window.exec_()

class ConfirmationWindow(QtWidgets.QDialog):
    def __init__(self, image_path,main_window, parent=None):
        super(ConfirmationWindow, self).__init__(parent)
        self.setWindowTitle("Crop Confirmation")
        self.image_path = image_path
        self.proceed = False
        self.main_window = main_window

        image_label = QtWidgets.QLabel()
        pixmap = QtGui.QPixmap(image_path)
        image_label.setPixmap(pixmap)
        image_label.setAlignment(QtCore.Qt.AlignCenter)

        proceed_button = QtWidgets.QPushButton("Save Image")
        proceed_button.clicked.connect(self.save_image)

        crop_again_button = QtWidgets.QPushButton("Crop Again")
        crop_again_button.clicked.connect(self.crop_again)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(proceed_button)
        button_layout.addWidget(crop_again_button)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(image_label)
        layout.addLayout(button_layout)

    def save_image(self):
        self.proceed = True
        self.accept()
        main_window.show_original_button()

    def crop_again(self):
        self.proceed = False
        self.accept()
        main_window.show_original_button()

class ScaleBarCalibrationWindow(QtWidgets.QDialog):
    def __init__(self, image_path, parent=None):
        super(ScaleBarCalibrationWindow, self).__init__(parent)
        self.setWindowTitle("ScaleBar Calibration")
        self.image_path = image_path
        self.image = None
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None
        self.scalebar_length_pixels = None

        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setScaledContents(True)  # Ensure image scales to fit label
        self.image_label.mousePressEvent = self.mousePressEvent
        self.image_label.mouseMoveEvent = self.mouseMoveEvent
        self.image_label.mouseReleaseEvent = self.mouseReleaseEvent

        self.scalebar_length_input = QtWidgets.QLineEdit()
        self.scalebar_length_input.setPlaceholderText("Enter scalebar length in micrometers")

        self.confirm_button = QtWidgets.QPushButton("Confirm")
        self.confirm_button.clicked.connect(self.calculate_pixels_in_scalebar)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.image_label)
        layout.addWidget(self.scalebar_length_input)
        layout.addWidget(self.confirm_button)

        self.load_image()

    def load_image(self):
        pixmap = QtGui.QPixmap(self.image_path)
        if not pixmap.isNull():
            # Adjust size of label to fit image
            label_width = min(800, pixmap.width())  # Adjust as needed for your application
            label_height = min(600, pixmap.height())  # Adjust as needed for your application
            scaled_pixmap = pixmap.scaled(label_width, label_height, QtCore.Qt.KeepAspectRatio)
            self.image_label.setPixmap(scaled_pixmap)
            self.image = scaled_pixmap.toImage()  # Use scaled image for calculations
        else:
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to load image: {self.image_path}")
            self.close()

    def mousePressEvent(self, event):
        pos = self.get_scaled_position(event.pos())
        self.start_x = pos.x()
        self.start_y = pos.y()
        self.end_x = pos.x()
        self.end_y = pos.y()
        self.update_overlay()

    def mouseMoveEvent(self, event):
        if self.start_x is not None and self.start_y is not None:
            pos = self.get_scaled_position(event.pos())
            self.end_x = pos.x()
            self.end_y = self.start_y  # Locks the y-coordinate to the initial click position
            self.update_overlay()

    def mouseReleaseEvent(self, event):
        pos = self.get_scaled_position(event.pos())
        self.end_x = pos.x()
        self.end_y = self.start_y  # Locks the y-coordinate to the initial click position
        self.update_overlay()

    def update_overlay(self):
        if self.image_label.pixmap() is None:
            return

        # Create an overlay pixmap of the same size as the original image
        overlay_pixmap = self.image_label.pixmap().copy()
        painter = QtGui.QPainter(overlay_pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        if self.start_x is not None and self.end_x is not None:
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 0, 0), 2))  # Red pen with 2 pixel width
            painter.drawLine(self.start_x, self.start_y, self.end_x, self.end_y)
        painter.end()
        self.image_label.setPixmap(overlay_pixmap)

    def get_scaled_position(self, pos):
        # Calculate the scale factors
        pixmap = self.image_label.pixmap()
        if not pixmap:
            return pos
        label_size = self.image_label.size()
        pixmap_size = pixmap.size()
        scale_factor_x = pixmap_size.width() / label_size.width()
        scale_factor_y = pixmap_size.height() / label_size.height()
        # Scale the positions accordingly
        return QtCore.QPoint(int(pos.x() * scale_factor_x), int(pos.y() * scale_factor_y))

    def calculate_pixels_in_scalebar(self):
        try:
            scalebar_length_um = float(self.scalebar_length_input.text())

            # Calculate length in pixels
            if self.start_x is not None and self.end_x is not None:
                length_pixels = abs(self.end_x - self.start_x)
                self.scalebar_length_pixels = length_pixels
                QtWidgets.QMessageBox.information(self, "Calibration Successful",
                                                  f"Scale bar length: {scalebar_length_um} µm\n"
                                                  f"Length in pixels: {length_pixels} pixels")
                self.accept()
            else:
                QtWidgets.QMessageBox.warning(self, "Error", "Please select a scale bar by drawing a line.")

        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Error", str(e))
            
    def get_calibrated_pixels(self):
        return self.scalebar_length_pixels
    
class PearliteLamellaOrientation2(QtWidgets.QWidget):
    def __init__(self, main_window=None, image_path=None):
        super(PearliteLamellaOrientation2, self).__init__(parent=main_window)
        self.main_window = main_window
        self.image_path = image_path
        self.setupUi()
        self.load_image(self.image_path)

    def setupUi(self):
        self.setObjectName("MainWindow")
        self.resize(1237, 856)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")

        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(90, 70, 700, 500))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        
        
        # self.label_image = CropWidget(self.frame)
        self.label_image=QtWidgets.QLabel(self.frame)
        self.label_image.setGeometry(QtCore.QRect(0, 0, self.frame.width(), self.frame.height()))
        self.label_image.setAlignment(QtCore.Qt.AlignCenter)
        self.label_image.setObjectName("label_image")
        self.label_image.setScaledContents(True)

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(180, 650, 161, 28))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setText('Scalebar Calibration')
        self.pushButton.clicked.connect(self.show_scalebar_calibration)

        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(400, 650, 230, 28))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.setText('Crop Image Without Scalebar')
        self.pushButton_2.clicked.connect(self.crop_image)
        self.original_button_text = self.pushButton_2.text()
        self.original_button_style = self.pushButton_2.styleSheet()

        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(890, 230, 200, 28))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.setText('Main Random Spacing')
        self.pushButton_3.clicked.connect(self.interlamellar_spacing)

        self.resultLabel = QtWidgets.QLabel(self.centralwidget)
        self.resultLabel.setGeometry(QtCore.QRect(890, 270, 300, 30))
        self.resultLabel.setObjectName("resultLabel")
        self.resultLabel.setText("Interlamellar spacing: ")

        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(890, 350, 200, 28))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_4.setText('Main Apparent Spacing')

        self.pushButton_6 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_6.setGeometry(QtCore.QRect(590, 720, 93, 28))
        self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton_6.setText('Next')
        
        self.setLayout(QtWidgets.QVBoxLayout())
        self.layout().addWidget(self.centralwidget)


    def load_image(self, image_path):
        print(f"Loading initial image from: {image_path}")
        global image
        image = cv2.imread(image_path, 0)
        if image is None:
            print("Failed to load initial image.")
            return
        
        self.display_image(self.label_image, image)
        
        print("Initial image loaded successfully.")
    
    
    def display_image(self, label, image):
        height, width = image.shape
        bytes_per_line = width
        q_img = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap.fromImage(q_img).scaled(label.size(), QtCore.Qt.KeepAspectRatio)
        label.setPixmap(pixmap)
    
    # def load_image(self,image_path):
    #     pixmap = QtGui.QPixmap(image_path)
    #     if not pixmap.isNull():
    #         label_width = min(self.frame.width(), pixmap.width())
    #         label_height = min(self.frame.height(), pixmap.height())
    #         scaled_pixmap = pixmap.scaled(label_width, label_height, QtCore.Qt.KeepAspectRatio)
    #         label_x = (self.frame.width() - label_width) // 2
    #         label_y = (self.frame.height() - label_height) // 2
    #         label_image = CropWidget(self.frame)
    #         label_image.setGeometry(label_x, label_y, label_width, label_height)
    #         label_image.setPixmap(scaled_pixmap)
    #         label_image.show()
    #     else:
    #         print("Failed to load image.")

    def crop_image(self):
        self.pushButton_2.setText("Crop now")
        self.pushButton_2.setStyleSheet("background-color: green; color: white;")
        self.crop_widget = CropWidget(self.frame)
        self.crop_widget.setGeometry(0, 0, self.frame.width(), self.frame.height())
        for widget in self.frame.findChildren(CropWidget):
            widget.enable_cropping()

    def show_original_button(self):
        # Revert the appearance of the button to its original state
        self.pushButton_2.setText(self.original_button_text)
        self.pushButton_2.setStyleSheet(self.original_button_style)

    def show_scalebar_calibration(self):
        self.calibration_window = ScaleBarCalibrationWindow(self.image_path)
        self.calibration_window.exec_()
    
    def get_cropped_image_path(self):
        for widget in self.frame.findChildren(CropWidget):
            if widget.cropped_image_path is not None:
                return widget.cropped_image_path
        return None

    def interlamellar_spacing(self):
        crop_image_path = self.get_cropped_image_path()
        if crop_image_path is None:
            QtWidgets.QMessageBox.warning(None, "Error", "Please crop the image first.")
            return
        
        scale_bar_length = self.calibration_window.scalebar_length_input.text()
        print("Scale bar length in pixels: ", scale_bar_length)
        calibrated_pixels = self.calibration_window.get_calibrated_pixels()
        print("Calibrated pixels: ", calibrated_pixels)
        img = cv2.imread(crop_image_path)

        interlamella_spacing_result = mrs.random_interlamellar_spacing(img, 200, int(calibrated_pixels), int(scale_bar_length))
        current_text = self.resultLabel.text()  # Get the current text in resultLabel
        new_text = f"{current_text} : {interlamella_spacing_result:.2f} μm"  # Append the spacing result
        self.resultLabel.setText(new_text)
        print(interlamella_spacing_result)
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())


