import os
import sys
import logging
import glob
import time
from tqdm import tqdm
import cv2
import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, MTCNN
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QLineEdit,
                             QCheckBox, QFileDialog, QMessageBox, QHBoxLayout, QDoubleSpinBox, QSpinBox, QSplitter,
                             QDockWidget, QProgressBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QSettings
from PyQt5.QtGui import QColor, QPalette, QPixmap, QImage
from PIL import Image

# Import PhotoViewer from Image_viewer_GUI (assuming it works perfectly)
from Image_viewer_GUI import PhotoViewer as ImageViewerPhotoViewer

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
logging.info("Starting application...")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(keep_all=True, device=device)




class MainWindow(QMainWindow):
    outputPathChanged = pyqtSignal(str)

    def __init__(self):
        super(MainWindow, self).__init__()
        self.photoViewer = None
        self.targetFaceImageLabel = QLabel()
        self.targetFaceImageLabel.setAlignment(Qt.AlignCenter)
        self.targetFaceImageLabel.setFixedSize(100, 100)
        self.targetFaceImageLabel.setStyleSheet("border: 1px solid black")
        self.settings = QSettings("YourOrganization", "FaceDetectionApp")
        self.setWindowTitle("Face Detection Configuration")
        self.setGeometry(100, 100, 1200, 768)
        self.targetFacePath = self.settings.value("targetFacePath", "") if os.path.exists(self.settings.value("targetFacePath", "")) else ""
        self.inputPath = self.settings.value("inputPath", "") if os.path.exists(self.settings.value("inputPath", "")) else ""
        self.exportPath = self.settings.value("exportPath", "") if os.path.exists(self.settings.value("exportPath", "")) else ""
        self.initUI()
        self.loadSettings()
        self.loadInitialTargetFaceImage()

    def initUI(self):
        mainSplitter = QSplitter(Qt.Horizontal)
        controlPanel = QWidget()
        controlLayout = QVBoxLayout()
        self.deviceCheckBox = QCheckBox("Use CUDA if available")
        self.deviceCheckBox.setChecked(True)
        controlLayout.addWidget(self.deviceCheckBox)

        # Batch Size SpinBox
        self.batchSizeSpinBox = QSpinBox()
        self.batchSizeSpinBox.setValue(200)  # Set initial value
        controlLayout.addWidget(self.createFormRow("Batch Size:", self.batchSizeSpinBox))

        self.resizeScaleSpinBox = QDoubleSpinBox()
        self.resizeScaleSpinBox.setSingleStep(0.05)
        self.resizeScaleSpinBox.setValue(0.35)
        controlLayout.addWidget(self.createFormRow("Resize Scale:", self.resizeScaleSpinBox))

        self.resizeScaleForSharpnessSpinBox = QDoubleSpinBox()
        self.resizeScaleForSharpnessSpinBox.setSingleStep(0.05)
        self.resizeScaleForSharpnessSpinBox.setValue(0.35)
        controlLayout.addWidget(self.createFormRow("Resize Scale for Sharpness:", self.resizeScaleForSharpnessSpinBox))

        self.sharpnessThresholdSpinBox = QSpinBox()
        self.sharpnessThresholdSpinBox.setMaximum(1000)
        self.sharpnessThresholdSpinBox.setValue(100)
        controlLayout.addWidget(self.createFormRow("Sharpness Threshold:", self.sharpnessThresholdSpinBox))

        self.minWidthSpinBox = QSpinBox()
        self.minWidthSpinBox.setMaximum(5000)
        self.minWidthSpinBox.setValue(500)
        self.minHeightSpinBox = QSpinBox()
        self.minHeightSpinBox.setMaximum(5000)
        self.minHeightSpinBox.setValue(500)
        minWidthHeightLayout = QHBoxLayout()
        minWidthHeightLayout.addWidget(self.createFormRow("Min Width:", self.minWidthSpinBox))
        minWidthHeightLayout.addWidget(self.createFormRow("Min Height:", self.minHeightSpinBox))
        controlLayout.addLayout(minWidthHeightLayout)

        self.marginFactorSpinBox = QDoubleSpinBox()
        self.marginFactorSpinBox.setSingleStep(0.1)
        self.marginFactorSpinBox.setValue(0.9)
        controlLayout.addWidget(self.createFormRow("Margin Factor:", self.marginFactorSpinBox))

        self.similarityThresholdSpinBox = QDoubleSpinBox()
        self.similarityThresholdSpinBox.setSingleStep(0.1)
        self.similarityThresholdSpinBox.setValue(0.9)
        controlLayout.addWidget(self.createFormRow("Similarity Threshold:", self.similarityThresholdSpinBox))

        self.extractFromVideoCheckBox = QCheckBox("Extract From Video")
        controlLayout.addWidget(self.extractFromVideoCheckBox)

        self.exportFullFrameCheckBox = QCheckBox("Export Full Frame")
        self.exportFullFrameCheckBox.setChecked(True)
        controlLayout.addWidget(self.exportFullFrameCheckBox)

        self.enableTargetFaceDetectionCheckBox = QCheckBox("Enable Target Face Detection")
        self.enableTargetFaceDetectionCheckBox.setChecked(True)
        controlLayout.addWidget(self.enableTargetFaceDetectionCheckBox)

        controlLayout.addWidget(self.targetFaceImageLabel)

        self.previewButton = QPushButton("Preview Results")
        controlLayout.addWidget(self.previewButton)
        self.previewButton.clicked.connect(self.previewResults)

        self.targetFacePathLineEdit = QLineEdit(self.settings.value("targetFacePath", ""))
        self.targetFacePathLineEdit.textChanged.connect(self.updateTargetFaceImageDisplay)
        controlLayout.addWidget(self.createPathFormRow("Target Face Path:", self.targetFacePathLineEdit, True))

        self.inputPathLineEdit = QLineEdit(self.settings.value("inputPath", ""))
        controlLayout.addWidget(self.createPathFormRow("Input Path:", self.inputPathLineEdit))

        self.exportPathLineEdit = QLineEdit(self.settings.value("exportPath", ""))
        controlLayout.addWidget(self.createPathFormRow("Export Path:", self.exportPathLineEdit))

        self.startButton = QPushButton("Start Detection")
        self.startButton.clicked.connect(self.startDetection)
        controlLayout.addWidget(self.startButton)

        self.stopButton = QPushButton("Stop")
        self.stopButton.setEnabled(False)
        controlLayout.addWidget(self.stopButton)
        self.stopButton.clicked.connect(self.stopDetection)

        # Create a progress bar
        self.progressBar = QProgressBar()
        self.progressBar.setValue(0)  # Set initial value to 0
        controlLayout.addWidget(self.progressBar)

        controlPanel.setLayout(controlLayout)
        mainSplitter.addWidget(controlPanel)
        self.setCentralWidget(mainSplitter) 

        # Create photo viewer widget
        self.photoViewer = ImageViewerPhotoViewer(self.exportPathLineEdit.text())
        photoViewerDockWidget = QDockWidget("Photo Viewer", self)
        photoViewerDockWidget.setWidget(self.photoViewer)
        self.addDockWidget(Qt.RightDockWidgetArea, photoViewerDockWidget)


        # Connect signals for dynamic updates
        self.resizeScaleSpinBox.valueChanged.connect(self.updateFaceDisplay)
        self.sharpnessThresholdSpinBox.valueChanged.connect(self.updateFaceDisplay)
        self.enableTargetFaceDetectionCheckBox.stateChanged.connect(self.updateFaceDisplay)
        self.targetFacePathLineEdit.textChanged.connect(self.updateFaceDisplay)
        self.marginFactorSpinBox.valueChanged.connect(self.updateFaceDisplay)
        self.similarityThresholdSpinBox.valueChanged.connect(self.updateFaceDisplay)
        self.resizeScaleForSharpnessSpinBox.valueChanged.connect(self.updateFaceDisplay)
        self.minWidthSpinBox.valueChanged.connect(self.updateFaceDisplay)
        self.minHeightSpinBox.valueChanged.connect(self.updateFaceDisplay)
        self.extractFromVideoCheckBox.stateChanged.connect(self.updateFaceDisplay)
        self.exportFullFrameCheckBox.stateChanged.connect(self.updateFaceDisplay)
        self.inputPathLineEdit.textChanged.connect(self.updateFaceDisplay)
        self.exportPathLineEdit.textChanged.connect(self.updateFaceDisplay)
        self.startButton.clicked.connect(self.updateFaceDisplay)
        self.stopButton.clicked.connect(self.updateFaceDisplay)

        self.loadInitialTargetFaceImage()
        self.updateFaceDisplay()

    @pyqtSlot(int)
    def updateProgressBar(self, value):
        self.progressBar.setValue(value)

    def setFolderPath(self, folderPath):
        self.folderPath = folderPath
        self.refresh()

    def loadInitialTargetFaceImage(self):
        targetFacePath = self.settings.value("targetFacePath", "")
        if targetFacePath and os.path.exists(targetFacePath):
            self.updateTargetFaceImageDisplay(targetFacePath)

    def updateTargetFaceImageDisplay(self, imagePath):
        if os.path.exists(imagePath):
            target_face_image = Image.open(imagePath)
            # Detect face
            boxes, _ = mtcnn.detect(target_face_image)
            if boxes is not None:
                # Get first face box
                box = boxes[0]
                # Apply margins
                margin = self.marginFactorSpinBox.value()
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                x1 -= w * margin
                y1 -= h * margin
                x2 += w * margin
                y2 += h * margin
                box = int(x1), int(y1), int(x2), int(y2)
                # Crop face with margins
                cropped_face = target_face_image.crop(box)

                # Convert PIL Image to QImage
                data = cropped_face.convert('RGBA').tobytes('raw', 'RGBA')
                qimage = QImage(data, cropped_face.size[0], cropped_face.size[1], QImage.Format_RGBA8888)

                # Create QPixmap from QImage
                qpixmap = QPixmap.fromImage(qimage)
                self.targetFaceImageLabel.setPixmap(qpixmap.scaled(self.targetFaceImageLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                self.targetFaceImageLabel.setText("No face detected")
        else:
            self.targetFaceImageLabel.setText("Image not available")

    def updateFaceDisplay(self):
        self.updateTargetFaceImageDisplay(self.targetFacePathLineEdit.text())

    def previewResults(self):
        outputFolderPath = self.exportPathLineEdit.text()
        if not outputFolderPath:
            QMessageBox.warning(self, "Warning", "Please select an output folder first.")
            return
        if self.photoViewer is None:
            self.photoViewer = ImageViewerPhotoViewer(outputFolderPath)
            self.photoViewer.show()
        else:
            self.photoViewer.setVisible(not self.photoViewer.isVisible())

    def loadSettings(self):
        self.targetFacePathLineEdit.setText(self.settings.value("targetFacePath", ""))
        self.inputPathLineEdit.setText(self.settings.value("inputPath", ""))
        self.exportPathLineEdit.setText(self.settings.value("exportPath", ""))
        self.loadInitialTargetFaceImage()

    def stopDetection(self):
        if self.detectionThread is not None and self.detectionThread.isRunning():
            self.detectionThread.terminate()
            self.detectionThread = None
            # Re-enable the start button and disable the stop button
            self.startButton.setEnabled(True)
            self.stopButton.setEnabled(False)
            QMessageBox.information(self, "Stopped", "Process Stop.")

    def startDetection(self):
        params = {
            'input_path': self.inputPathLineEdit.text(),
            'export_path': self.exportPathLineEdit.text(),
            'target_face_path': self.targetFacePathLineEdit.text() if self.enableTargetFaceDetectionCheckBox.isChecked() else None,
            'resize_scale': self.resizeScaleSpinBox.value(),
            'resize_scale_for_sharpness': self.resizeScaleForSharpnessSpinBox.value(),
            'sharpness_threshold': self.sharpnessThresholdSpinBox.value(),
            'min_width': self.minWidthSpinBox.value(),
            'min_height': self.minHeightSpinBox.value(),
            'export_full_frame': self.exportFullFrameCheckBox.isChecked(),
            'margin_factor': self.marginFactorSpinBox.value(),
            'enable_target_face_detection': self.enableTargetFaceDetectionCheckBox.isChecked(),
            'extract_from_video': self.extractFromVideoCheckBox.isChecked(),
            'device': device,
            'similarity_threshold': self.similarityThresholdSpinBox.value(),
            'batch_size': self.batchSizeSpinBox.value(),
        }
        if self.enableTargetFaceDetectionCheckBox.isChecked():
            params['target_face_path'] = self.targetFacePathLineEdit.text()
        else:
            params['target_face_path'] = None
        self.progressBar.setValue(0)  # Reset progress bar
        self.detectionThread = DetectionThread(params)
        self.detectionThread.progress.connect(self.updateProgressBar)
        self.detectionThread.finished.connect(self.onDetectionFinished)
        self.detectionThread.start()
        self.startButton.setEnabled(False)
        self.stopButton.setEnabled(True)

    @pyqtSlot(bool, str)
    def onDetectionFinished(self, success, message):
        QMessageBox.information(self, "Detection Status", message)
        # Re-enable the start button and disable the stop button
        self.startButton.setEnabled(True)
        self.stopButton.setEnabled(False)

    def createFormRow(self, label, widget):
        row = QWidget()
        rowLayout = QHBoxLayout()
        row.setLayout(rowLayout)
        rowLayout.addWidget(QLabel(label))
        rowLayout.addWidget(widget)
        return row

    def createPathFormRow(self, label, widget, selectFile=False):
        rowWidget = QWidget()
        rowLayout = QHBoxLayout()
        rowWidget.setLayout(rowLayout)
        rowLayout.addWidget(QLabel(label))
        rowLayout.addWidget(widget)
        browseButton = QPushButton("Browse")
        browseButton.clicked.connect(lambda: self.openFileDialog(widget, selectFile))
        rowLayout.addWidget(browseButton)
        return rowWidget

    def openFileDialog(self, lineEdit, selectFile=False):
        options = QFileDialog.Options()
        if selectFile:
            filePaths, _ = QFileDialog.getOpenFileNames(self, "Select Files", "", "Images (*.png *.jpg *.jpeg *.mp4 *.avi *.mkv)", options=options)
            if filePaths:
                lineEdit.setText(';'.join(filePaths))  # Join multiple file paths with semicolons
        else:
            directory = QFileDialog.getExistingDirectory(self, "Select Directory", "", options=options)
            if directory:
                lineEdit.setText(directory)
                if lineEdit == self.inputPathLineEdit:
                    self.settings.setValue("inputPath", directory)
                    self.inputPath = directory
                elif lineEdit == self.exportPathLineEdit:
                    self.settings.setValue("exportPath", directory)
                    self.exportPath = directory
    def validateParameters(params):
        if params.get('enable_target_face_detection', False) and not params.get('target_face_path'):
            logging.error("Target face detection is enabled, but no target face path is provided.")
            return False
        return True

    def validatePaths(self):
        if self.enableTargetFaceDetectionCheckBox.isChecked() and not os.path.exists(self.targetFacePath):
            QMessageBox.warning(self, "Invalid Path", "The target face path is invalid. Please select a valid path.")
            return False
        if not os.path.exists(self.inputPath):
            QMessageBox.warning(self, "Invalid Path", "The input path is invalid. Please select a valid path.")
            return False
        if not os.path.exists(self.exportPath):
            try:
                os.makedirs(self.exportPath)
            except Exception as e:
                QMessageBox.warning(self, "Path Creation Failed", f"Failed to create the export path: {e}")
                return False
        return True


class DetectionThread(QThread):
    finished = pyqtSignal(bool, str)
    imageReady = pyqtSignal(str)
    progress = pyqtSignal(int)
    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            # Extract parameters
            input_path = self.params['input_path']
            export_path = self.params['export_path']
            resize_scale = self.params['resize_scale']
            resize_scale_for_sharpness = self.params['resize_scale_for_sharpness']
            sharpness_threshold = self.params['sharpness_threshold']
            min_width = self.params['min_width']
            min_height = self.params['min_height']
            export_full_frame = self.params['export_full_frame']
            margin_factor = self.params['margin_factor']
            enable_target_face_detection = self.params['enable_target_face_detection']
            extract_from_video = self.params['extract_from_video']
            target_face_path = self.params.get('target_face_path', None)
            similarity_threshold = self.params['similarity_threshold']
            batch_size = self.params['batch_size']

            # Initialize target_face_embedding
            target_face_embedding = None
            if enable_target_face_detection and target_face_path:
                try:
                    target_face_image = Image.open(target_face_path)
                    target_face_image = self._detect_and_crop_face(target_face_image, mtcnn, margin_factor)
                    target_face_embedding = self._get_embedding(model, target_face_image)
                    logging.info("Target face embedding loaded.")
                except Exception as e:
                    logging.error(f"Failed to load target face embedding: {e}")
                    self.finished.emit(False, "Failed to load target face embedding.")
                    return

            total_items = 0
            items_processed = 0

            if os.path.isdir(input_path):
                filenames = glob.glob(os.path.join(input_path, '*.*'))
                total_items += len(filenames)

                for filename in tqdm(filenames, desc="Processing Files", unit="file"):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image = cv2.imread(filename)
                        if image is not None:
                            self._process_image(image, filename, mtcnn, export_path, resize_scale, resize_scale_for_sharpness,
                                                sharpness_threshold, min_width, min_height, export_full_frame, margin_factor,
                                                enable_target_face_detection, target_face_embedding, similarity_threshold)
                            items_processed += 1
                            self.progress.emit(int(items_processed / total_items * 100))
                    elif extract_from_video and filename.lower().endswith(('.mp4', '.avi', '.mkv')):
                        items_processed += self._process_video(filename, mtcnn, export_path, resize_scale, resize_scale_for_sharpness,
                                                                sharpness_threshold, min_width, min_height, export_full_frame, margin_factor,
                                                                enable_target_face_detection, target_face_embedding, similarity_threshold,
                                                                batch_size, total_items, items_processed)
                        self.progress.emit(int(items_processed / total_items * 100))

            elif os.path.isfile(input_path):
                total_items = 1  # Since it's a single file, total_items is set to 1
                if input_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image = cv2.imread(input_path)
                    if image is not None:
                        self._process_image(image, input_path, mtcnn, export_path, resize_scale, resize_scale_for_sharpness,
                                            sharpness_threshold, min_width, min_height, export_full_frame, margin_factor,
                                            enable_target_face_detection, target_face_embedding, similarity_threshold)
                        items_processed = 1
                        self.progress.emit(100)  # Since it's a single file, directly set progress to 100%
                elif extract_from_video and input_path.lower().endswith(('.mp4', '.avi', '.mkv')):
                    items_processed = self._process_video(input_path, mtcnn, export_path, resize_scale, resize_scale_for_sharpness,
                                                        sharpness_threshold, min_width, min_height, export_full_frame, margin_factor,
                                                        enable_target_face_detection, target_face_embedding, similarity_threshold,
                                                        batch_size, total_items, 0)
                    self.progress.emit(100)  # Since it's a single file, directly set progress to 100%

            self.finished.emit(True, "Detection completed successfully.")
            self.imageReady.emit(export_path)
        except Exception as e:
            logging.error(f"Detection thread encountered an error: {e}")
            self.finished.emit(False, "Detection failed.")

    def _run_detection(self, input_path, export_path, resize_scale, resize_scale_for_sharpness, sharpness_threshold,
                    min_width, min_height, export_full_frame, margin_factor, enable_target_face_detection,
                    extract_from_video, target_face_path, similarity_threshold, batch_size):
        target_face_embedding = None
        if enable_target_face_detection:
            if target_face_path is None or not os.path.exists(target_face_path):
                logging.error("Target face detection is enabled, but no valid target face path was provided.")
                return
            else:
                target_face_image = Image.open(target_face_path)
                target_face_image = self._detect_and_crop_face(target_face_image, mtcnn, margin_factor)
                target_face_embedding = self._get_embedding(model, target_face_image)
                logging.info("Target face embedding loaded.")
        else:
            logging.info("Target face detection is disabled. Processing all detected faces without similarity check.")

        total_items = 0
        items_processed = 0

        if os.path.isdir(input_path):
            filenames = glob.glob(os.path.join(input_path, '*.*'))
            total_items += len(filenames)

            for filename in tqdm(filenames, desc="Processing Files", unit="file"):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image = cv2.imread(filename)
                    if image is not None:
                        self._process_image(image, filename, mtcnn, export_path, resize_scale, resize_scale_for_sharpness,
                                            sharpness_threshold, min_width, min_height, export_full_frame, margin_factor,
                                            enable_target_face_detection, target_face_embedding, similarity_threshold)
                        items_processed += 1
                        self.progress.emit(int(items_processed / total_items * 100))
                elif extract_from_video and filename.lower().endswith(('.mp4', '.avi', '.mkv')):
                    items_processed += self._process_video(filename, mtcnn, export_path, resize_scale, resize_scale_for_sharpness,
                                                        sharpness_threshold, min_width, min_height, export_full_frame, margin_factor,
                                                        enable_target_face_detection, target_face_embedding, similarity_threshold,
                                                        batch_size, total_items, items_processed)
                    self.progress.emit(int(items_processed / total_items * 100))

        elif extract_from_video and os.path.isfile(input_path) and input_path.lower().endswith(('.mp4', '.avi', '.mkv')):
            total_items += 1
            items_processed += self._process_video(input_path, mtcnn, export_path, resize_scale, resize_scale_for_sharpness,
                                                sharpness_threshold, min_width, min_height, export_full_frame, margin_factor,
                                                enable_target_face_detection, target_face_embedding, similarity_threshold,
                                                batch_size, total_items, items_processed)
            self.progress.emit(int(items_processed / total_items * 100))            

    def _get_embedding(self, model, face_image):
        transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor()
        ])
        if face_image.mode == 'RGBA':
            face_image = face_image.convert('RGB')
        face_image = transform(face_image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model(face_image)
        return embedding

    def _is_target_face(self, detected_face_embedding, target_face_embedding, threshold):
        dist = torch.nn.functional.pairwise_distance(detected_face_embedding, target_face_embedding)
        logging.info(f"Distance: {dist.item()}")
        return dist.item() < threshold

    def _is_blurry(self, image, threshold):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()
        return fm < threshold

    def _process_image(self, image, filename, mtcnn, export_path, resize_scale, resize_scale_for_sharpness, sharpness_threshold,
                        min_width, min_height, export_full_frame, margin_factor, enable_target_face_detection,
                        target_face_embedding, similarity_threshold):
        logging.info(f"Processing image: {filename}")
        original_image = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (0, 0), fx=resize_scale, fy=resize_scale)
        logging.info("Detecting faces...")
        boxes, _ = mtcnn.detect([image])
        if boxes is not None and boxes[0] is not None:
            logging.info(f"Detected {len(boxes[0])} faces in {filename}")
            for box in boxes[0]:
                if box is not None:
                    logging.info(f"Original Box: {box}")
                    width = box[2] - box[0]
                    height = box[3] - box[1]
                    margin_x = width * margin_factor / 2
                    margin_y = height * margin_factor / 2
                    x1 = max(0, int((box[0] - margin_x) / resize_scale))
                    y1 = max(0, int((box[1] - margin_y) / resize_scale))
                    x2 = min(original_image.shape[1], int((box[2] + margin_x) / resize_scale))
                    y2 = min(original_image.shape[0], int((box[3] + margin_y) / resize_scale))
                    logging.info(f"Adjusted Box: [{x1}, {y1}, {x2}, {y2}]")
                    if x2 <= x1 or y2 <= y1:
                        logging.info("Invalid box dimensions, skipping box.")
                        continue
                    cropped_face = original_image[y1:y2, x1:x2]
                    if cropped_face.size == 0 or cropped_face.shape[1] < min_width or cropped_face.shape[0] < min_height:
                        logging.info("Face too small, skipping.")
                        continue
                    resized_face_for_sharpness = cv2.resize(cropped_face, (0, 0), fx=resize_scale_for_sharpness,
                                                           fy=resize_scale_for_sharpness)
                    # Perform sharpness check if sharpness_threshold is not False
                    if sharpness_threshold is not False and self._is_blurry(resized_face_for_sharpness, sharpness_threshold):
                        logging.info("Face is blurry, skipping.")
                        continue
                    cropped_face_pil = Image.fromarray(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))
                    detected_face_embedding = self._get_embedding(model, cropped_face_pil)
                    if enable_target_face_detection and target_face_embedding is not None:
                        if self._is_target_face(detected_face_embedding, target_face_embedding, similarity_threshold):
                            logging.info(f"Target face detected in {filename}.")
                            self._save_face(original_image, cropped_face, export_path, filename, export_full_frame)
                        else:
                            logging.info(f"No target face match in {filename}.")
                    else:
                        self._save_face(original_image, cropped_face, export_path, filename, export_full_frame)
        else:
            logging.info(f"No faces detected in {filename}.")

    def _process_video(self, video_path, mtcnn, export_path, resize_scale, resize_scale_for_sharpness, sharpness_threshold,
                    min_width, min_height, export_full_frame, margin_factor, enable_target_face_detection,
                    target_face_embedding, similarity_threshold, batch_size, total_items, items_processed):
        v_cap = cv2.VideoCapture(video_path)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        original_frames = []
        frames_processed = 0

        for j in tqdm(range(v_len), total=v_len, desc="Processing Video", unit="frame"):
            success, frame = v_cap.read()
            if not success:
                continue
            original_frames.append(frame.copy())
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)
            frames.append(frame)

            if len(frames) >= batch_size or j == v_len - 1:
                boxes, _ = mtcnn.detect(frames)
                frames_processed += len(frames)
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        if box is not None:
                            for b in box:
                                logging.info(f"Original Box: {b}")
                                x1, y1, x2, y2 = [max(0, int(b[k] / resize_scale)) for k in range(4)]
                                width = x2 - x1
                                height = y2 - y1
                                margin_x = width * margin_factor / 2
                                margin_y = height * margin_factor / 2
                                x1 = max(0, x1 - int(margin_x))
                                y1 = max(0, y1 - int(margin_y))
                                x2 = min(original_frames[i].shape[1], x2 + int(margin_x))
                                y2 = min(original_frames[i].shape[0], y2 + int(margin_y))
                                logging.info(f"Adjusted Box: [{x1}, {y1}, {x2}, {y2}]")
                                if x2 <= x1 or y2 <= y1:
                                    logging.info("Invalid box dimensions, skipping box.")
                                    continue
                                cropped_face = original_frames[i][y1:y2, x1:x2]
                                if cropped_face.size == 0 or cropped_face.shape[1] < min_width or cropped_face.shape[0] < min_height:
                                    continue
                                resized_face_for_sharpness = cv2.resize(cropped_face, (0, 0), fx=resize_scale_for_sharpness,
                                                                    fy=resize_scale_for_sharpness)
                                if self._is_blurry(resized_face_for_sharpness, sharpness_threshold):
                                    continue
                                cropped_face_pil = Image.fromarray(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))
                                detected_face_embedding = self._get_embedding(model, cropped_face_pil)
                                if enable_target_face_detection:
                                    if self._is_target_face(detected_face_embedding, target_face_embedding, similarity_threshold):
                                        logging.info(f"Target face detected in frame {frames_processed + i}.")
                                        self._save_face(original_frames[i], cropped_face, export_path,
                                                        f'frame_{frames_processed + i}', export_full_frame)
                                    else:
                                        logging.info(f"No target face match in frame {frames_processed + i}.")
                                else:
                                    self._save_face(original_frames[i], cropped_face, export_path,
                                                    f'frame_{frames_processed + i}', export_full_frame)
                frames = []
                original_frames = []  # Release original frames to avoid memory buildup

                # Update progress
                items_processed += frames_processed
                progress_value = int(items_processed / total_items * 100)

        v_cap.release()
        return frames_processed

    def _save_face(self, original_image, cropped_face, export_path, filename, export_full_frame):
        base_filename = os.path.splitext(os.path.basename(filename))[0]
        if export_full_frame:
            save_path = os.path.join(export_path, f'{base_filename}.png')
        else:
            save_path = os.path.join(export_path, f'face_{base_filename}.png')
        Image.fromarray(cv2.cvtColor(cropped_face if not export_full_frame else original_image, cv2.COLOR_BGR2RGB)).save(save_path)
        logging.info(f"Saved face to {save_path}")

    def _detect_and_crop_face(self, image, mtcnn, margin_factor):
        image_rgb = image.convert('RGB')
        boxes, _ = mtcnn.detect(image_rgb)
        if boxes is not None:
            box = boxes[0]
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            x1 -= w * margin_factor
            y1 -= h * margin_factor
            x2 += w * margin_factor
            y2 += h * margin_factor
            box = int(x1), int(y1), int(x2), int(y2)
            return image_rgb.crop(box)
        return image_rgb


def applyDarkTheme(app):
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)



def main():
    app = QApplication(sys.argv)
    applyDarkTheme(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()