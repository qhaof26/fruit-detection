# import tkinter as tk
# from tkinter import filedialog, messagebox
# from PIL import Image, ImageTk
# import cv2
# import torch
# from ultralytics import YOLO
#
# class FruitDetectionApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Fruit Detection & Validation")
#         self.root.geometry("1200x700")
#
#         # Khởi tạo các thành phần giao diện
#         self.panel = tk.Label(root)
#         self.panel.pack(padx=10, pady=10)
#
#         self.status_label = tk.Label(root, text="Trạng thái: Chưa tải model", font=("Arial", 14))
#         self.status_label.pack(pady=10)
#
#         self.load_model_btn = tk.Button(root, text="Tải Model", command=self.load_model)
#         self.load_model_btn.pack(side=tk.LEFT, padx=10)
#
#         self.load_image_btn = tk.Button(root, text="Tải Ảnh", command=self.detect_fruits)
#         self.load_image_btn.pack(side=tk.LEFT, padx=10)
#
#         self.model = None
#         self.current_image = None
#
#     def load_model(self):
#         try:
#             # Load mô hình từ file best.pt (đảm bảo file best.pt nằm cùng thư mục với script)
#             model_path = "best.pt"
#             self.model = YOLO(model_path)
#             messagebox.showinfo("Success", "Model loaded successfully!")
#         except Exception as e:
#             messagebox.showerror("Error", f"Failed to load model: {str(e)}")
#
#     def detect_fruits(self):
#         if not self.model:
#             messagebox.showwarning("Cảnh báo", "Vui lòng tải model trước!")
#             return
#
#         image_path = filedialog.askopenfilename(
#             title="Chọn ảnh để nhận diện",
#             filetypes=[("Image files", "*.jpg *.jpeg *.png")]
#         )
#
#         if not image_path:
#             return
#
#         try:
#             # Đọc ảnh gốc
#             self.current_image = cv2.imread(image_path)
#             original_image = self.current_image.copy()
#
#             # Dự đoán
#             results = self.model(original_image, conf=0.4)  # Ngưỡng tự tin 0.4
#
#             # Xử lý kết quả
#             class_counts = {}
#             valid_classes = []
#             invalid_classes = []
#
#             for result in results:
#                 boxes = result.boxes.xyxy.cpu().numpy()
#                 confs = result.boxes.conf.cpu().numpy()
#                 class_ids = result.boxes.cls.cpu().numpy()
#
#                 # Lọc các detection có confidence >= 0.4
#                 filtered_indices = confs >= 0.4
#                 boxes = boxes[filtered_indices]
#                 confs = confs[filtered_indices]
#                 class_ids = class_ids[filtered_indices]
#
#                 # Đếm số lượng từng loại quả
#                 for cls_id in class_ids:
#                     class_name = self.model.names[int(cls_id)]
#                     class_counts[class_name] = class_counts.get(class_name, 0) + 1
#
#                 # Xác định lớp chiếm đa số
#                 if class_counts:
#                     majority_class = max(class_counts, key=class_counts.get)
#                     majority_count = class_counts[majority_class]
#
#                     # Kiểm tra các lớp thiểu số
#                     for cls_id in class_ids:
#                         class_name = self.model.names[int(cls_id)]
#                         if class_counts[class_name] < majority_count:
#                             invalid_classes.append((class_name, cls_id))
#                         else:
#                             valid_classes.append((class_name, cls_id))
#
#                     # Vẽ bounding box
#                     for box, conf, cls_id in zip(boxes, confs, class_ids):
#                         x1, y1, x2, y2 = map(int, box)
#                         class_name = self.model.names[int(cls_id)]
#
#                         # Màu sắc dựa trên hợp lệ/không hợp lệ
#                         color = (0, 255, 0)  # Xanh lá nếu hợp lệ
#                         if class_name in [c[0] for c in invalid_classes]:
#                             color = (0, 0, 255)  # Đỏ nếu không hợp lệ
#
#                         # Vẽ bounding box và label
#                         cv2.rectangle(original_image, (x1, y1), (x2, y2), color, 5)
#                         label = f"{class_name} {conf:.2f}"
#                         (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 4, 4)
#                         cv2.rectangle(original_image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
#                         cv2.putText(original_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 4)
#
#             # Hiển thị trạng thái
#             if not class_counts:
#                 self.status_label.config(text="Trạng thái: Không phát hiện được trái cây hợp lệ", fg="red")
#             else:
#                 if len(class_counts) == 1:
#                     self.status_label.config(text=f"Trạng thái: HỢP LỆ", fg="green")
#                 else:
#                     self.status_label.config(text=f"Trạng thái: KHÔNG HỢP LỆ", fg="red")
#
#             # Hiển thị ảnh trong giao diện
#             image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
#             img = Image.fromarray(image_rgb)
#             img.thumbnail((1000, 600))
#             img_tk = ImageTk.PhotoImage(img)
#             self.panel.configure(image=img_tk)
#             self.panel.image = img_tk
#
#         except Exception as e:
#             self.status_label.config(text=f"Lỗi: {str(e)}", fg="red")
#             messagebox.showerror("Lỗi", f"Không thể xử lý ảnh: {str(e)}")
#
# if __name__ == "__main__":
#     root = tk.Tk()
#     app = FruitDetectionApp(root)
#     root.mainloop()
#=======================================================================================================================
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton,
                             QVBoxLayout, QHBoxLayout, QWidget, QFileDialog,
                             QMessageBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from ultralytics import YOLO
from collections import Counter


class FruitDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fruit Detection YOLOv8l")
        self.setGeometry(100, 100, 1000, 700)

        # Load YOLOv8 model
        self.load_model()

        # Initialize variables
        self.image_path = None
        self.processed_image = None

        # Setup UI
        self.setup_ui()

    def load_model(self):
        try:
            # Load YOLOv8 model
            self.model = YOLO("best.pt")
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

        # Detect button
        self.detect_button = QPushButton("Detect Fruits")
        self.detect_button.clicked.connect(self.detect_fruits)
        self.detect_button.setEnabled(False)
        button_layout.addWidget(self.detect_button)

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
            self.detect_button.setEnabled(True)
            self.result_label.setText("Image loaded. Click 'Detect Fruits' to process.")

    def display_image(self, image_path=None, processed=False):
        if processed and self.processed_image is not None:
            # Display processed image
            height, width, channel = self.processed_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(self.processed_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        elif image_path:
            # Display original image
            pixmap = QPixmap(image_path)
            q_image = pixmap.toImage()
        else:
            return

        # Scale image to fit the label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(q_image)
        pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(),
                               Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)

    def detect_fruits(self):
        if not self.image_path:
            return

        try:
            # Load image
            image = cv2.imread(self.image_path)
            original_image = image.copy()

            # Run YOLOv8 detection
            results = self.model(image)

            # Get detection results
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()

            # Get class names from the model
            class_names = results[0].names

            # Ánh xạ tên tiếng Anh sang tiếng Việt
            vietnamese_names = {
                'Apple': 'Táo',
                'Orange': 'Cam',
                'Banana': 'Chuối',
                'Guava': 'Ổi'
            }

            # Count occurrences of each fruit type
            detected_classes = [class_names[int(cls)] for cls in classes]
            class_counts = Counter(detected_classes)

            # Determine the main fruit (with highest count)
            if class_counts:
                main_fruit = class_counts.most_common(1)[0][0]

                # Draw bounding boxes with appropriate colors
                for i, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
                    x1, y1, x2, y2 = map(int, box)
                    class_name = class_names[int(cls)]

                    # Green for main fruit, red for others
                    color = (0, 255, 0) if class_name == main_fruit else (0, 0, 255)

                    # Vẽ bounding box dày hơn
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

                    # Tên tiếng Việt của loại quả
                    vn_class_name = vietnamese_names.get(class_name, class_name)

                    # Add label with class name and confidence
                    label = f"{vn_class_name}: {conf:.2f}"

                    # Calculate text size for better positioning
                    (text_width, text_height), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
                    )

                    # Draw background rectangle for text
                    cv2.rectangle(
                        image,
                        (x1, y1 - text_height - 10),
                        (x1 + text_width + 10, y1),
                        color,
                        -1
                    )

                    # Add text with larger font and thicker lines
                    cv2.putText(
                        image,
                        label,
                        (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,  # Tăng kích thước font
                        (255, 255, 255),
                        2  # Tăng độ dày của chữ
                    )

                # Convert BGR to RGB for display
                self.processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.display_image(processed=True)

                # Check if all fruits are of the same type
                is_valid = len(class_counts) == 1

                # Display results in Vietnamese
                main_fruit_vn = vietnamese_names.get(main_fruit, main_fruit)
                result_text = f"Quả chính: {main_fruit_vn} (Số lượng: {class_counts[main_fruit]})\n"

                for fruit, count in class_counts.items():
                    if fruit != main_fruit:
                        fruit_vn = vietnamese_names.get(fruit, fruit)
                        result_text += f"Quả khác loại: {fruit_vn}: {count}\n"

                status = "Hợp lệ" if is_valid else "Không hợp lệ"
                self.result_label.setText(f"{result_text}\nTrạng thái: {status}")

                # Set color based on validity
                self.result_label.setStyleSheet(
                    "font-size: 16pt; font-weight: bold; color: green;" if is_valid
                    else "font-size: 16pt; font-weight: bold; color: red;"
                )
            else:
                self.result_label.setText("Không phát hiện trái cây trong ảnh.")
                self.result_label.setStyleSheet("font-size: 16pt; font-weight: bold; color: black;")

        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Lỗi trong quá trình phát hiện: {e}")
            print(f"Lỗi: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FruitDetectionApp()
    window.show()
    sys.exit(app.exec_())