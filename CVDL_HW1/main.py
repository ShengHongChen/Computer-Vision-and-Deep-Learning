from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QPixmap
from ui import Ui_MainWindow

import sys
import os
import numpy
import cv2
import matplotlib.pyplot as plt

from PIL import Image

class main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        self.ui.pushButton0_1.clicked.connect(self.load_folder)
        self.ui.pushButton0_2.clicked.connect(self.load_image_L)
        self.ui.pushButton0_3.clicked.connect(self.load_image_R)
        self.ui.pushButton1_1.clicked.connect(self.find_corners)

        self.folder_path = None
        self.image_L_path = None
        self.image_R_path = None
        
        self.criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)
        self.width = 11
        self.height = 8

    def load_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.folder_path = folder_path
            print(f"Folder loaded: {self.folder_path}")

    def load_image_L(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open Left Image",
            "",
            "Images (*.png *.xpm *.jpg *.bmp);;All Files (*)",
            options=options
        )
        if file_name:
            self.image_L_path = file_name
            print(f"Left image loaded: {self.image_L_path}")

    def load_image_R(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open Right Image",
            "",
            "Images (*.png *.xpm *.jpg *.bmp);;All Files (*)",
            options=options
        )
        if file_name:
            self.image_R_path = file_name
            print(f"Right image loaded: {self.image_R_path}")

    def get_folder_path(self):
        return self.folder_path

    def get_image_L_path(self):
        return self.image_L_path

    def get_image_R_path(self):
        return self.image_R_path

    def find_corners(self):
        if self.folder_path is None:
            print("Please load folder first!")
            return
        
        image_files = sorted([f for f in os.listdir(self.folder_path) if f.endswith('.bmp')])
        
        for image_file in image_files:
            image_path = os.path.join(self.folder_path, image_file)
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            ret, corners = cv2.findChessboardCorners(gray, (self.width, self.height), None)
            
            if ret:
                corners2 = cv2.cornerSubPix(
                    gray, 
                    corners, 
                    (5, 5),
                    (-1, -1),
                    self.criteria
                )
                cv2.drawChessboardCorners(img, (self.width, self.height), corners2, ret)
                
                scale_percent = 40
                width = int(img.shape[1] * scale_percent / 100)
                height = int(img.shape[0] * scale_percent / 100)
                dim = (width, height)
                resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

                cv2.imshow(f'Corners - {image_file}', resized_img)
                cv2.waitKey(500)
                
        cv2.waitKey(0)
        cv2.destroyAllWindows()

app = QApplication([])
window = main()
window.show()
sys.exit(app.exec_())
