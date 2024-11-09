from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from ui import Ui_MainWindow

import sys
import os
import numpy
import cv2
import matplotlib.pyplot as plt


class main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        self.ui.pushButton0_1.clicked.connect(self.load_folder)
        self.ui.pushButton0_2.clicked.connect(self.load_image_L)
        self.ui.pushButton0_3.clicked.connect(self.load_image_R)
        self.ui.pushButton1_1.clicked.connect(self.find_corners)
        self.ui.pushButton1_2.clicked.connect(self.find_intrinsic)
        self.ui.pushButton1_3.clicked.connect(self.find_extrinsic)
        self.ui.pushButton1_4.clicked.connect(self.find_distortion)
        self.ui.pushButton1_5.clicked.connect(self.show_undistorted)
        self.ui.pushButton2_1.clicked.connect(self.show_words_on_board)
        self.ui.pushButton2_2.clicked.connect(self.show_words_vertically)
        self.ui.pushButton3_1.clicked.connect(self.stereo_disparity_map)
        self.ui.pushButton4_1.clicked.connect(self.load_image_1)
        self.ui.pushButton4_2.clicked.connect(self.load_image_2)
        self.ui.pushButton4_3.clicked.connect(self.find_keypoints)
        self.ui.pushButton4_4.clicked.connect(self.find_matched_keypoints)

        self.folder_path = None
        self.image_L_path = None
        self.image_R_path = None
        
        self.criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)
        self.width = 11
        self.height = 8

        self.objpoints = []
        self.imgpoints = []
        self.objp = numpy.zeros((self.height * self.width, 3), numpy.float32)
        self.objp[:, :2] = numpy.mgrid[0:self.width, 0:self.height].T.reshape(-1, 2) * 0.02

        self.ui.spinBox.setMinimum(1)
        self.ui.spinBox.setMaximum(15)

        self.image_1_path = None
        self.image_2_path = None

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
        
        self.objpoints = []
        self.imgpoints = []
        
        image_files = sorted([f for f in os.listdir(self.folder_path) if f.endswith('.bmp')])
        
        for image_file in image_files:
            image_path = os.path.join(self.folder_path, image_file)
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            ret, corners = cv2.findChessboardCorners(gray, (self.width, self.height), None)
            
            if ret:
                self.objpoints.append(self.objp)
                corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), self.criteria)
                self.imgpoints.append(corners2)
                
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

    def find_intrinsic(self):
        if not hasattr(self, 'imgpoints') or len(self.imgpoints) == 0:
            print("Please find corners first!")
            return
            
        img_size = (2048, 2048)
        
        ret, ins, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, 
            self.imgpoints, 
            img_size, 
            None, 
            None
        )
        
        self.ins = ins
        self.dist = dist
        self.rvecs = rvecs
        self.tvecs = tvecs
        
        plt.figure(figsize=(8, 6))
        plt.text(0.1, 0.5, f'Intrinsic:\n{ins}', fontsize=12, family='monospace')
        plt.axis('off')
        plt.title('Intrinsic Matrix')
        plt.show()

    def find_extrinsic(self):
        if not hasattr(self, 'rvecs') or not hasattr(self, 'tvecs'):
            print("Please find intrinsic matrix first!")
            return

        image_number = self.ui.spinBox.value() - 1
        
        rotation_matrix, _ = cv2.Rodrigues(self.rvecs[image_number])
        
        translation_vector = self.tvecs[image_number].reshape(3, 1)
        
        extrinsic_matrix = numpy.hstack((rotation_matrix, translation_vector))
        
        plt.figure(figsize=(10, 8))
        plt.text(0.1, 0.5, 
                f'Extrinsic Matrix of image {image_number + 1}:\n\n{extrinsic_matrix}', 
                fontsize=12, 
                family='monospace')
        plt.axis('off')
        plt.title(f'Extrinsic Matrix (Image {image_number + 1})')
        plt.show()

    def find_distortion(self):
        if not hasattr(self, 'dist'):
            print("Please find intrinsic matrix first!")
            return

        image_number = self.ui.spinBox.value() - 1
        
        plt.figure(figsize=(8, 6))
        plt.text(0.1, 0.5, 
                f'Distortion Matrix:\n[k1, k2, p1, p2, k3] =\n{self.dist[0]}', 
                fontsize=12, 
                family='monospace')
        plt.axis('off')
        plt.title(f'Distortion Coefficients (Image {image_number + 1})')
        plt.show()

    def show_undistorted(self):
        if not hasattr(self, 'ins') or not hasattr(self, 'dist'):
            print("Please find intrinsic matrix first!")
            return
        
        if self.folder_path is None:
            print("Please load folder first!")
            return

        image_number = self.ui.spinBox.value() - 1
        
        image_files = sorted([f for f in os.listdir(self.folder_path) if f.endswith('.bmp')])
        if not image_files:
            print("No images found in folder!")
            return

        img_path = os.path.join(self.folder_path, image_files[image_number])
        img = cv2.imread(img_path)
        
        undistorted_img = cv2.undistort(img, self.ins, self.dist)
        
        plt.figure(figsize=(15, 6))
        
        plt.subplot(121)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f'Original Image {image_number + 1}')
        plt.axis('off')
        
        plt.subplot(122)
        plt.imshow(cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB))
        plt.title(f'Undistorted Image {image_number + 1}')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

    def show_words_on_board(self):
        if self.folder_path is None:
            print("Please load folder first!")
            return

        if not hasattr(self, 'ins') or not hasattr(self, 'dist') or not hasattr(self, 'rvecs') or not hasattr(self, 'tvecs'):
            print("Calibrating camera...")
            
            self.objpoints = []
            self.imgpoints = []
            
            image_files = sorted([f for f in os.listdir(self.folder_path) if f.endswith('.bmp')])
            for image_file in image_files:
                image_path = os.path.join(self.folder_path, image_file)
                img = cv2.imread(image_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                ret, corners = cv2.findChessboardCorners(gray, (self.width, self.height), None)
                if ret:
                    self.objpoints.append(self.objp)
                    corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), self.criteria)
                    self.imgpoints.append(corners2)
            
            img_size = (2048, 2048)
            ret, self.ins, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(
                self.objpoints, 
                self.imgpoints, 
                img_size, 
                None, 
                None
            )
            print("Camera calibration completed.")

        input_text = self.ui.textEdit.toPlainText().upper()
        if not input_text or len(input_text) > 6:
            print("Please input 1-6 characters!")
            return

        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(script_dir, 'Dataset_CvDl_Hw1', 'Q2_Image', 'Q2_db', 'alphabet_db_onboard.txt')
            
            print(f"Reading database file: {db_path}")
            
            fs = cv2.FileStorage(db_path, cv2.FILE_STORAGE_READ)
            if not fs.isOpened():
                print("Failed to open database file.")
                return

            predefined_positions = [
                (8, 5, 0),
                (5, 5, 0),
                (2, 5, 0),
                (8, 2, 0),
                (5, 2, 0),
                (2, 2, 0)
            ]
            
            image_files = sorted([f for f in os.listdir(self.folder_path) if f.endswith('.bmp')])
            
            for idx, image_file in enumerate(image_files):
                img_path = os.path.join(self.folder_path, image_file)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Cannot read image {image_file}")
                    continue
                
                img_copy = img.copy()

                for i, letter in enumerate(input_text):
                    char_node = fs.getNode(letter)
                    if char_node.empty():
                        print(f"Character {letter} not found in database")
                        continue
                    
                    try:
                        char_points = char_node.mat()
                        if char_points is None:
                            continue
                            
                        char_points = char_points.reshape(-1, 3).astype(numpy.float64)
                        scale = 0.02
                        char_points *= scale
                        
                        char_points += numpy.array(predefined_positions[i]) * 0.02
                        
                        points_2d, _ = cv2.projectPoints(
                            char_points,
                            self.rvecs[idx],
                            self.tvecs[idx],
                            self.ins,
                            self.dist
                        )
                        
                        for j in range(0, len(points_2d)-1, 2):
                            pt1 = tuple(points_2d[j][0].astype(int))
                            pt2 = tuple(points_2d[j+1][0].astype(int))
                            cv2.line(img_copy, pt1, pt2, (0, 0, 255), 3)
                            
                    except Exception as e:
                        print(f"Error processing character {letter}: {str(e)}")
                        continue
                
                try:
                    scale_percent = 40
                    width = int(img_copy.shape[1] * scale_percent / 100)
                    height = int(img_copy.shape[0] * scale_percent / 100)
                    resized_img = cv2.resize(img_copy, (width, height), interpolation=cv2.INTER_AREA)
                    
                    cv2.imshow(f'Image {idx + 1}', resized_img)
                    cv2.waitKey(0)
                    cv2.destroyWindow(f'Image {idx + 1}')
                    
                except Exception as e:
                    print(f"Error displaying image: {str(e)}")
            
        except Exception as e:
            print(f"Error: {str(e)}")
            
        finally:
            if 'fs' in locals():
                fs.release()

    def show_words_vertically(self):
        if self.folder_path is None:
            print("Please load folder first!")
            return

        if not hasattr(self, 'ins') or not hasattr(self, 'dist') or not hasattr(self, 'rvecs') or not hasattr(self, 'tvecs'):
            print("Calibrating camera...")
            
            self.objpoints = []
            self.imgpoints = []
            
            image_files = sorted([f for f in os.listdir(self.folder_path) if f.endswith('.bmp')])
            for image_file in image_files:
                image_path = os.path.join(self.folder_path, image_file)
                img = cv2.imread(image_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                ret, corners = cv2.findChessboardCorners(gray, (self.width, self.height), None)
                if ret:
                    self.objpoints.append(self.objp)
                    corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), self.criteria)
                    self.imgpoints.append(corners2)
            
            img_size = (2048, 2048)
            ret, self.ins, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(
                self.objpoints, 
                self.imgpoints, 
                img_size, 
                None, 
                None
            )
            print("Camera calibration completed.")

        input_text = self.ui.textEdit.toPlainText().upper()
        if not input_text or len(input_text) > 6:
            print("Please input 1-6 characters!")
            return

        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(script_dir, 'Dataset_CvDl_Hw1', 'Q2_Image', 'Q2_db', 'alphabet_db_vertical.txt')
            
            print(f"Reading database file: {db_path}")
            
            fs = cv2.FileStorage(db_path, cv2.FILE_STORAGE_READ)
            if not fs.isOpened():
                print("Failed to open database file.")
                return

            predefined_positions = [
                (8, 5, 0),
                (5, 5, 0),
                (2, 5, 0),
                (8, 2, 0),
                (5, 2, 0),
                (2, 2, 0)
            ]
            
            image_files = sorted([f for f in os.listdir(self.folder_path) if f.endswith('.bmp')])
            
            for idx, image_file in enumerate(image_files):
                img_path = os.path.join(self.folder_path, image_file)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Cannot read image {image_file}")
                    continue
                
                img_copy = img.copy()
                
                for i, letter in enumerate(input_text):
                    char_node = fs.getNode(letter)
                    if char_node.empty():
                        print(f"Character {letter} not found in database")
                        continue
                    
                    try:
                        char_points = char_node.mat()
                        if char_points is None:
                            continue
                            
                        char_points = char_points.reshape(-1, 3).astype(numpy.float64)
                        scale = 0.02
                        char_points *= scale
                        
                        char_points += numpy.array(predefined_positions[i]) * 0.02
                        
                        points_2d, _ = cv2.projectPoints(
                            char_points,
                            self.rvecs[idx],
                            self.tvecs[idx],
                            self.ins,
                            self.dist
                        )
                        
                        for j in range(0, len(points_2d)-1, 2):
                            pt1 = tuple(points_2d[j][0].astype(int))
                            pt2 = tuple(points_2d[j+1][0].astype(int))
                            cv2.line(img_copy, pt1, pt2, (0, 0, 255), 3)
                            
                    except Exception as e:
                        print(f"Error processing character {letter}: {str(e)}")
                        continue
                
                try:
                    scale_percent = 40
                    width = int(img_copy.shape[1] * scale_percent / 100)
                    height = int(img_copy.shape[0] * scale_percent / 100)
                    resized_img = cv2.resize(img_copy, (width, height), interpolation=cv2.INTER_AREA)
                    
                    cv2.imshow(f'Image {idx + 1}', resized_img)
                    cv2.waitKey(0)
                    cv2.destroyWindow(f'Image {idx + 1}')
                    
                except Exception as e:
                    print(f"Error displaying image: {str(e)}")
            
        except Exception as e:
            print(f"Error: {str(e)}")
            
        finally:
            if 'fs' in locals():
                fs.release()

    def stereo_disparity_map(self):
        if self.image_L_path is None or self.image_R_path is None:
            print("Please load both left and right images first!")
            return
        
        imgL = cv2.imread(self.image_L_path, cv2.IMREAD_GRAYSCALE)
        imgR = cv2.imread(self.image_R_path, cv2.IMREAD_GRAYSCALE)
        
        if imgL is None or imgR is None:
            print("Error loading images!")
            return
        
        stereo = cv2.StereoBM.create(numDisparities=432, blockSize=25)
        
        disparity = stereo.compute(imgL, imgR)
        
        normalized_disparity = cv2.normalize(
            disparity, 
            None, 
            alpha=0, 
            beta=255, 
            norm_type=cv2.NORM_MINMAX, 
            dtype=cv2.CV_8U
        )
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        plt.imshow(imgL, cmap='gray')
        plt.title('Left Image')
        plt.axis('off')
        
        plt.subplot(132)
        plt.imshow(imgR, cmap='gray')
        plt.title('Right Image')
        plt.axis('off')
        
        plt.subplot(133)
        plt.imshow(normalized_disparity, cmap='gray')
        plt.title('Disparity Map')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

    def load_image_1(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image 1",
            "",
            "Images (*.png *.xpm *.jpg *.bmp);;All Files (*)",
            options=options
        )
        if file_name:
            self.image_1_path = file_name
            print(f"Image 1 loaded: {self.image_1_path}")

    def load_image_2(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image 2",
            "",
            "Images (*.png *.xpm *.jpg *.bmp);;All Files (*)",
            options=options
        )
        if file_name:
            self.image_2_path = file_name
            print(f"Image 2 loaded: {self.image_2_path}")

    def find_keypoints(self):
        if self.image_1_path is None:
            print("Please load Image 1 first!")
            return
        
        img = cv2.imread(self.image_1_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        
        img_keypoints = cv2.drawKeypoints(
            gray, 
            keypoints, 
            None, 
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(img_keypoints, cv2.COLOR_BGR2RGB))
        plt.title('SIFT Keypoints')
        plt.axis('off')
        plt.show()

    def find_matched_keypoints(self):
        if self.image_1_path is None or self.image_2_path is None:
            print("Please load both images first!")
            return
        
        img1 = cv2.imread(self.image_1_path)
        img2 = cv2.imread(self.image_2_path)
        
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
        
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)
        
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append([m])
            
        img_matches = cv2.drawMatchesKnn(
            img1, keypoints1,
            img2, keypoints2,
            good_matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        plt.figure(figsize=(12, 6))
        plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        plt.title('Matched Keypoints')
        plt.axis('off')
        plt.show()

app = QApplication([])
window = main()
window.show()
sys.exit(app.exec_())
