import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton,
                             QVBoxLayout, QHBoxLayout, QWidget, QFileDialog,
                             QMessageBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import tensorflow as tf
from keras.api.preprocessing.image import img_to_array
from keras.src.applications.densenet import preprocess_input

class FruitClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fruit Detection DenseNet201")
        self.setGeometry(100, 100, 800, 600)

        # Load model
        self.load_model()

        # Initialize variables
        self.image_path = None
        self.fruit_classes = ['Apple', 'Banana', 'Guava', 'Orange']

        # Setup UI
        self.setup_ui()

    def load_model(self):
        try:
            # Load DenseNet model
            self.densenet_model = tf.keras.models.load_model("best_densenet.h5")
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load model: {e}")
            sys.exit(1)

    def setup_ui(self):
        # Main layout
        main_layout = QVBoxLayout()

        # Button layout
        button_layout = QHBoxLayout()

        # Load image button
        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)
        button_layout.addWidget(self.load_button)

        # Classify button
        self.classify_button = QPushButton("Classify Fruit")
        self.classify_button.clicked.connect(self.classify_fruit)
        self.classify_button.setEnabled(False)
        button_layout.addWidget(self.classify_button)

        main_layout.addLayout(button_layout)

        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("No image loaded")
        self.image_label.setStyleSheet("border: 1px solid black")
        main_layout.addWidget(self.image_label)

        # Result label
        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setText("Results will appear here")
        self.result_label.setStyleSheet("font-size: 16pt; font-weight: bold;")
        main_layout.addWidget(self.result_label)

        # Confidence scores layout
        scores_layout = QVBoxLayout()
        self.score_labels = []

        for fruit in self.fruit_classes:
            label = QLabel(f"{fruit}: 0%")
            label.setAlignment(Qt.AlignLeft)
            scores_layout.addWidget(label)
            self.score_labels.append(label)

        main_layout.addLayout(scores_layout)

        # Set central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def load_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")

        if file_path:
            self.image_path = file_path
            self.display_image(file_path)
            self.classify_button.setEnabled(True)
            self.result_label.setText("Image loaded. Click 'Classify Fruit' to process.")

            # Reset score labels
            for i, fruit in enumerate(self.fruit_classes):
                self.score_labels[i].setText(f"{fruit}: 0%")

    def display_image(self, image_path):
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)

    def classify_fruit(self):
        if not self.image_path:
            return

        try:
            # Load and preprocess image
            image = cv2.imread(self.image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Resize to 224x224 (DenseNet input size)
            image_resized = cv2.resize(image, (224, 224))

            # Convert to array and preprocess
            image_array = img_to_array(image_resized)
            image_preprocessed = preprocess_input(image_array)
            image_batch = np.expand_dims(image_preprocessed, axis=0)

            # Get predictions
            predictions = self.densenet_model.predict(image_batch, verbose=0)

            # Get the predicted class and confidence
            predicted_class_index = np.argmax(predictions[0])
            predicted_class = self.fruit_classes[predicted_class_index]
            confidence = predictions[0][predicted_class_index] * 100

            # Update result label
            self.result_label.setText(f"Predicted: {predicted_class} ({confidence:.2f}%)")

            # Update confidence scores for all classes
            for i, fruit in enumerate(self.fruit_classes):
                score = predictions[0][i] * 100
                self.score_labels[i].setText(f"{fruit}: {score:.2f}%")

                # Highlight the predicted class
                if i == predicted_class_index:
                    self.score_labels[i].setStyleSheet("color: green; font-weight: bold;")
                else:
                    self.score_labels[i].setStyleSheet("color: black; font-weight: normal;")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during classification: {e}")
            print(f"Error: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FruitClassifierApp()
    window.show()
    sys.exit(app.exec_())