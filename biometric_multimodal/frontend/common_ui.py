from __future__ import annotations

import json
from urllib.request import Request, urlopen
from urllib.error import URLError

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QGridLayout,
    QVBoxLayout,
    QTextEdit,
    QLineEdit,
    QHBoxLayout,
    QSizePolicy,
)


class ImagePathPicker(QWidget):
    def __init__(self, label_text: str, preview_size: tuple[int, int] = (220, 220)):
        super().__init__()
        self.preview_width, self.preview_height = preview_size

        self.label = QLabel(label_text)
        self.value = QLineEdit()
        self.value.setReadOnly(True)

        self.button = QPushButton("Browse")
        self.button.clicked.connect(self.browse)

        self.preview = QLabel("No image selected")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setFixedSize(self.preview_width, self.preview_height)
        self.preview.setStyleSheet(
            "border: 1px solid #999; background-color: #f5f5f5; color: #555;"
        )
        self.preview.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        top_row = QHBoxLayout()
        top_row.addWidget(self.label)
        top_row.addWidget(self.value)
        top_row.addWidget(self.button)

        layout = QVBoxLayout(self)
        layout.addLayout(top_row)
        layout.addWidget(self.preview)

    def browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if path:
            self.value.setText(path)
            self.update_preview(path)

    def update_preview(self, path: str):
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self.preview.setText("Preview unavailable")
            return

        scaled = pixmap.scaled(
            self.preview_width,
            self.preview_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview.setPixmap(scaled)

    def text(self) -> str:
        return self.value.text().strip()


class SingleModalityWindow(QWidget):
    def __init__(self, title: str, endpoint: str, enrollment_label: str, probe_label: str):
        super().__init__()
        self.setWindowTitle(title)
        self.endpoint = endpoint

        self.subject_id = QLineEdit()
        self.subject_id.setPlaceholderText("Claimed subject id")

        self.enrollment = ImagePathPicker(enrollment_label)
        self.probe = ImagePathPicker(probe_label)

        self.run_btn = QPushButton("Verify")
        self.run_btn.clicked.connect(self.run_verification)

        self.output = QTextEdit()
        self.output.setReadOnly(True)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Claimed subject"))
        layout.addWidget(self.subject_id)
        layout.addWidget(self.enrollment)
        layout.addWidget(self.probe)
        layout.addWidget(self.run_btn)
        layout.addWidget(self.output)

    def run_verification(self):
        payload = {
            "subject_id": self.subject_id.text().strip(),
            "enrollment": self.enrollment.text(),
            "probe": self.probe.text(),
        }

        data = json.dumps(payload).encode("utf-8")
        req = Request(
            self.endpoint,
            data=data,
            headers={"Content-Type": "application/json"}
        )

        try:
            with urlopen(req) as resp:
                self.output.setText(resp.read().decode("utf-8"))
        except URLError as e:
            self.output.setText(f"Backend error: {e}")