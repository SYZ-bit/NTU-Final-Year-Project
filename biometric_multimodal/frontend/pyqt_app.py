from __future__ import annotations

import json
import sys
from urllib.request import Request, urlopen
from urllib.error import URLError

from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QTextEdit,
    QLineEdit,
    QGridLayout,
)

from frontend.common_ui import ImagePathPicker


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multimodal Biometrics")

        self.subject_id = QLineEdit()
        self.subject_id.setPlaceholderText("Claimed subject id")

        self.en_face = ImagePathPicker("Enrollment face")
        self.en_fp = ImagePathPicker("Enrollment fingerprint")
        self.en_palm = ImagePathPicker("Enrollment palm")

        self.pr_face = ImagePathPicker("Probe face")
        self.pr_fp = ImagePathPicker("Probe fingerprint")
        self.pr_palm = ImagePathPicker("Probe palm")

        self.run_btn = QPushButton("Verify")
        self.run_btn.clicked.connect(self.run_verification)

        self.output = QTextEdit()
        self.output.setReadOnly(True)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(QLabel("Claimed subject"))
        main_layout.addWidget(self.subject_id)

        grid = QGridLayout()
        grid.addWidget(self.en_face, 0, 0)
        grid.addWidget(self.en_fp, 0, 1)
        grid.addWidget(self.en_palm, 0, 2)
        grid.addWidget(self.pr_face, 1, 0)
        grid.addWidget(self.pr_fp, 1, 1)
        grid.addWidget(self.pr_palm, 1, 2)

        main_layout.addLayout(grid)
        main_layout.addWidget(self.run_btn)
        main_layout.addWidget(self.output)

    def run_verification(self):
        payload = {
            "subject_id": self.subject_id.text().strip(),
            "enrollment": {
                "face": self.en_face.text(),
                "fingerprint": self.en_fp.text(),
                "palm": self.en_palm.text(),
            },
            "probe": {
                "face": self.pr_face.text(),
                "fingerprint": self.pr_fp.text(),
                "palm": self.pr_palm.text(),
            },
        }

        data = json.dumps(payload).encode("utf-8")
        req = Request(
            "http://127.0.0.1:5000/verify",
            data=data,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urlopen(req) as resp:
                self.output.setText(resp.read().decode("utf-8"))
        except URLError as e:
            self.output.setText(f"Backend error: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1200, 900)
    win.show()
    sys.exit(app.exec())