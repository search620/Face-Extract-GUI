import sys
import os
import time
import logging
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QFileDialog, QWidget, QLabel, QPushButton, QScrollArea, QGridLayout, QSlider, QCheckBox, QDialog, QHBoxLayout)
from PyQt5.QtCore import Qt, QSize, QTimer, QFileSystemWatcher, QThreadPool, QRunnable, pyqtSignal, QObject
from PyQt5.QtGui import QPixmap, QColor, QPalette
import multiprocessing
import cProfile
import pstats
from PIL import Image
import io
from PIL import Image, ImageOps
from PyQt5.QtWidgets import QListView, QStyledItemDelegate
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtCore import QSize
import time

logging.basicConfig(level=logging.WARNING)

profile = cProfile.Profile()
profile.enable() 

cpu_count = multiprocessing.cpu_count()
QThreadPool.globalInstance().setMaxThreadCount(cpu_count)


class SignalEmitter(QObject):
    imageLoaded = pyqtSignal(QPixmap, str, int, str)
    
    

class ImageLoaderTask(QRunnable):
    def __init__(self, imagePath, emitter, index, folderPath, targetSize=None, retries=5, retry_delay=5):
        super(ImageLoaderTask, self).__init__()
        self.imagePath = imagePath
        self.emitter = emitter
        self.index = index
        self.folderPath = folderPath
        self.targetSize = targetSize  # Target size is expected to be a tuple (width, height)
        self._stopped = False
        self.retries = retries
        self.retry_delay = retry_delay

    def stop(self):
        self._stopped = True

    def run(self):
        if self._stopped:
            return

        retries = self.retries
        while retries > 0:
            try:
                with Image.open(self.imagePath) as img:
                    img.info.pop('icc_profile', None)  # Remove the ICC profile if it exists

                    # Only resize if necessary and maintain aspect ratio
                    if self.targetSize is not None:
                        img.thumbnail(self.targetSize, Image.ANTIALIAS)

                    bytes_io = io.BytesIO()
                    img_format = 'JPEG' if img.mode == 'RGB' else 'PNG'
                    img.save(bytes_io, format=img_format)
                    pixmap = QPixmap()
                    pixmap.loadFromData(bytes_io.getvalue())
                    self.emitter.imageLoaded.emit(pixmap, self.imagePath, self.index, self.folderPath)
                    return  # Image loaded successfully, exit the loop
            except Exception as e:
                logging.error(f"Failed to load image {self.imagePath}: {e}")
                retries -= 1
                if retries > 0:
                    logging.info(f"Retrying loading image {self.imagePath} in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)

        # If all retries failed, emit a signal or log an error message
        logging.error(f"Failed to load image {self.imagePath} after {self.retries} retries.")


class ImagePopup(QDialog):
    def __init__(self, parent=None):
        super(ImagePopup, self).__init__(parent, Qt.FramelessWindowHint | Qt.Popup)
        self.layout = QVBoxLayout(self)
        self.label = QLabel(self)
        self.layout.addWidget(self.label)
        self.label.setAlignment(Qt.AlignCenter)

    def setPixmap(self, pixmap):
        self.label.setPixmap(pixmap)

    def mousePressEvent(self, event):
        self.hide()
        self.parent().popup_closed()

class HoverLabel(QLabel):
    def __init__(self, pixmap, parent=None):
        super(HoverLabel, self).__init__(parent)
        self.popup_recently_closed = False
        self.original_pixmap = pixmap
        self.setPixmap(pixmap.scaled(parent.current_image_size, parent.current_image_size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.hover_enabled = False
        self.parent_widget = parent
        self.popup = ImagePopup(self)
        self.show_timer = QTimer(self)
        self.show_timer.setSingleShot(True)
        self.show_timer.setInterval(550)
        self.show_timer.timeout.connect(self.showPopup)
        self.mouse_inside = False

    def popup_closed(self):
        self.popup_recently_closed = True
        QTimer.singleShot(100, lambda: setattr(self, 'popup_recently_closed', False))

    def enterEvent(self, event):
        if self.hover_enabled and not self.popup_recently_closed:
            self.show_timer.start()
        super(HoverLabel, self).enterEvent(event)

    def leaveEvent(self, event):
        self.show_timer.stop()
        super(HoverLabel, self).leaveEvent(event)

    def showPopup(self):
        max_width = self.parent_widget.width() * 1
        max_height = self.parent_widget.height() * 1
        scaled_pixmap = self.original_pixmap.scaled(max_width, max_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.popup.setPixmap(scaled_pixmap)
        self.popup.resize(scaled_pixmap.size())
        popup_position = self.parent_widget.geometry().center() - self.popup.rect().center()
        popup_position.setX(max(popup_position.x(), 0))
        popup_position.setY(max(popup_position.y(), 0))
        self.popup.move(popup_position)
        self.popup.show()

class PhotoViewer(QMainWindow):
    def __init__(self, folderPath=None):
        super().__init__()
        self.setWindowTitle("Real-Time Photo Viewer")
        self.setGeometry(100, 100, 800, 800)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.updateTimer = QTimer(self)
        self.updateTimer.setSingleShot(True)
        self.updateTimer.timeout.connect(self.actualLoadImages)
        self.updateTimer.setInterval(1000)

        self.resizeTimer = QTimer(self)
        self.resizeTimer.setSingleShot(True)
        self.resizeTimer.timeout.connect(self.onResizeTimeout)
        self.resizeTimer.setInterval(500)  

        self.signalEmitter = SignalEmitter()
        self.signalEmitter.imageLoaded.connect(self.displayImage)


        self.threadPool = QThreadPool.globalInstance()
        cpu_count = multiprocessing.cpu_count()
        self.threadPool.setMaxThreadCount(cpu_count)
        self.runningTasks = []
        self.image_files = []
        self.loaded_images = {}
        self.batch_size = 0
        self.current_batch = 0        

        self.threadPool = QThreadPool()

        self.current_image_size = 200
        self.scroll_area = QScrollArea()
        self.scroll_area_widget = QWidget()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area_layout = QGridLayout(self.scroll_area_widget)
        self.scroll_area_layout.setSpacing(10)
        self.scroll_area.setWidget(self.scroll_area_widget)
        self.layout.addWidget(self.scroll_area)


        controlsLayout = QHBoxLayout()
        self.hover_checkbox = QCheckBox("Hover On")
        controlsLayout.addWidget(self.hover_checkbox)
        self.layout.addLayout(controlsLayout)
        self.imageCounterLabel = QLabel("Images: 0 / 0")
        controlsLayout.addWidget(self.imageCounterLabel, 1) 
        self.themeToggleButton = QPushButton("Toggle Theme")
        self.themeToggleButton.setFixedSize(120, 30)
        controlsLayout.addWidget(self.themeToggleButton)



        self.folder_watcher = QFileSystemWatcher(self)
        self.folder_watcher.directoryChanged.connect(self.loadImages)

        self.selected_folder = folderPath
        if self.selected_folder:
            self.folder_watcher.addPath(self.selected_folder)
            self.loadImages()

        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_folder)
        self.layout.addWidget(self.browse_button)

        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setMinimum(100)
        self.size_slider.setMaximum(400)
        self.size_slider.setValue(200)
        self.size_slider.sliderReleased.connect(self.applySliderChange)
        self.layout.addWidget(self.size_slider)

        self.hover_checkbox.stateChanged.connect(self.updateHoverEffect)
        self.themeToggleButton.clicked.connect(self.toggleTheme)

        self.isDarkTheme = False
        self.applyDarkTheme()

        self.image_files = []
   
    def applyDarkTheme(self):
        app = QApplication.instance()
        if app is not None:
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

    def applyLightTheme(self):
        app = QApplication.instance()
        if app is not None:
            app.setStyle("Fusion")
            app.setPalette(app.style().standardPalette())

    def toggleTheme(self):
        if self.isDarkTheme:
            self.applyLightTheme()
        else:
            self.applyDarkTheme()
        self.isDarkTheme = not self.isDarkTheme
   
    def updateImageCounter(self):
        # Update the text to show the number of loaded images out of the total number of image files
        self.imageCounterLabel.setText(f"Images: {len(self.loaded_images)} / {len(self.image_files)}")

    def updateHoverEffect(self):
        enabled = self.hover_checkbox.isChecked()
        for i in range(self.scroll_area_layout.count()):
            widget = self.scroll_area_layout.itemAt(i).widget()
            if isinstance(widget, QWidget):
                label = widget.findChild(HoverLabel)
                if label:
                    label.hover_enabled = enabled
                    label.show_timer.stop()

    def applySliderChange(self):
        self.current_image_size = self.size_slider.value()
        self.displayImages()

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.selected_folder = folder
            self.folder_watcher.addPath(folder)

            # Introduce a delay before loading images
            self.loadImagesTimer = QTimer()
            self.loadImagesTimer.setSingleShot(True)
            self.loadImagesTimer.timeout.connect(self.loadImages)
            self.loadImagesTimer.start(500) 
    def loadImages(self):
        self.updateTimer.start()

    def actualLoadImages(self):
        # Get the current list of image files in the directory
        existing_files = set([os.path.join(self.selected_folder, f) for f in os.listdir(self.selected_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        current_files = set(self.image_files)
        
        # Determine new and removed files
        new_files = existing_files - current_files
        removed_files = current_files - existing_files
        
        # Remove display for deleted images
        for imagePath in removed_files:
            self.removeImageDisplay(imagePath)
        
        # Load new images
        for imagePath in new_files:
            index = len(self.image_files)  # Assign a new index for the image
            self.image_files.append(imagePath)  # Add new image path to the list
            task = ImageLoaderTask(imagePath, self.signalEmitter, index, self.selected_folder, targetSize=None)
            self.runningTasks.append(task)  # Keep track of the task
            self.threadPool.start(task)
        
        # Update GUI after changes
        QTimer.singleShot(100, self.displayImages)
    def removeImageDisplay(self, imagePath):
        # Find the index of the image to be removed
        index = self.image_files.index(imagePath)

        # Remove the image path from the list
        self.image_files.remove(imagePath)

        # Remove the corresponding widget if it exists
        if index in self.loaded_images:
            widget = self.loaded_images.pop(index)
            self.scroll_area_layout.removeWidget(widget)
            widget.deleteLater()
            self.updateImageCounter()

        # Adjust indexes of the remaining loaded images
        new_loaded_images = {}
        for old_index, widget in self.loaded_images.items():
            if old_index > index:
                new_index = old_index - 1
                new_loaded_images[new_index] = widget
            else:
                new_loaded_images[old_index] = widget
        self.loaded_images = new_loaded_images

        # Schedule a display update after a short delay
        QTimer.singleShot(100, self.displayImages)

    def displayImage(self, pixmap, imagePath, index, folderPath):
        if folderPath != self.selected_folder:
            return

        if index not in self.loaded_images:
            label = HoverLabel(pixmap, self)
            label.hover_enabled = self.hover_checkbox.isChecked()
            
            current_column_count = max(1, self.scroll_area.width() // (self.current_image_size + self.scroll_area_layout.spacing()))
            row = index // current_column_count
            col = index % current_column_count
            
            image_container = QWidget()
            container_layout = QVBoxLayout(image_container)
            container_layout.addStretch()
            container_layout.addWidget(label)
            container_layout.setAlignment(label, Qt.AlignBottom | Qt.AlignCenter)
            container_layout.setContentsMargins(5, 5, 5, 5)
            
            self.scroll_area_layout.addWidget(image_container, row, col)
            self.loaded_images[index] = label

        self.updateImageCounter()


    def displayImages(self):
        if not self.image_files:
            return

        self.clearLayout(self.scroll_area_layout)
        row, col = 0, 0
        max_col = max(1, self.scroll_area.width() // (self.current_image_size + self.scroll_area_layout.spacing()))

        for index, image_path in enumerate(self.image_files):
            if index in self.loaded_images:
                label = self.loaded_images[index]
                label.setPixmap(label.original_pixmap.scaled(self.current_image_size, self.current_image_size, Qt.KeepAspectRatio, Qt.SmoothTransformation))

                image_container = QWidget()
                container_layout = QVBoxLayout(image_container)
                container_layout.addStretch()
                container_layout.addWidget(label)
                container_layout.setAlignment(label, Qt.AlignBottom | Qt.AlignCenter)
                container_layout.setContentsMargins(5, 5, 5, 5)
                self.scroll_area_layout.addWidget(image_container, row // max_col, row % max_col)
                row += 1

        self.scroll_area_widget.setLayout(self.scroll_area_layout)

    def loadNextBatch(self):
        start_index = self.current_batch * self.batch_size
        end_index = min(start_index + self.batch_size, len(self.image_files))
        for i in range(start_index, end_index):
            if i not in self.loaded_images:
                imagePath = self.image_files[i]
                task = ImageLoaderTask(imagePath, self.signalEmitter, i, (100, 100))
                self.threadPool.start(task)
        self.current_batch += 1

    def resizeEvent(self, event):
        super(PhotoViewer, self).resizeEvent(event)
        self.displayImages()

    def onResizeTimeout(self):
        self.displayImages()

    def clearLayout(self, layout):
        for i in reversed(range(layout.count())): 
            widget = layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
            layout.removeItem(layout.itemAt(i))



if __name__ == "__main__":
    with cProfile.Profile() as pr:
        app = QApplication(sys.argv)
        window = PhotoViewer()
        window.show()
        exit_status = app.exec_()
        pr.disable()
        
        with open("profiling_results.txt", "w") as f:
            stats = pstats.Stats(pr, stream=f)
            stats.sort_stats('cumtime')
            stats.print_stats()
        
        sys.exit(exit_status)
