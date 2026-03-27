from __future__ import annotations

import json
import sys
from pathlib import Path

import requests
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSizePolicy,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

API_URL = "http://127.0.0.1:5000/verify"
IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"


def score_to_percent(score: float | None) -> int:
    if score is None:
        return 0
    score = max(0.0, min(1.0, float(score)))
    return int(score * 100)


class ImageInputCard(QGroupBox):
    def __init__(self, title: str, placeholder: str):
        super().__init__(title)

        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText(placeholder)

        self.preview = QLabel("No image selected")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setFixedSize(340, 260)
        self.preview.setFrameShape(QFrame.Shape.Box)
        self.preview.setStyleSheet("""
            QLabel {
                background-color: #f7f7f7;
                border: 1px solid #cfcfcf;
            }
        """)

        self.meta_label = QLabel("Path: -")
        self.meta_label.setWordWrap(True)

        browse_btn = QPushButton("Browse")
        clear_btn = QPushButton("Clear")

        browse_btn.clicked.connect(self.browse_file)
        clear_btn.clicked.connect(self.clear)

        row = QHBoxLayout()
        row.addWidget(self.path_input)
        row.addWidget(browse_btn)
        row.addWidget(clear_btn)

        layout = QVBoxLayout()
        layout.addLayout(row)
        layout.addWidget(self.preview, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.meta_label)
        self.setLayout(layout)

    def browse_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select image", "", IMAGE_FILTER)
        if path:
            self.set_path(path)

    def set_path(self, path: str):
        self.path_input.setText(path)
        self.meta_label.setText(f"Path: {path}")
        self.load_preview(path)

    def load_preview(self, path: str):
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self.preview.setPixmap(QPixmap())
            self.preview.setText("Preview unavailable")
            return

        scaled = pixmap.scaled(
            self.preview.width(),
            self.preview.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview.setText("")
        self.preview.setPixmap(scaled)

    def clear(self):
        self.path_input.clear()
        self.preview.setPixmap(QPixmap())
        self.preview.setText("No image selected")
        self.meta_label.setText("Path: -")

    def get_path(self) -> str | None:
        value = self.path_input.text().strip()
        return value if value else None


class MetricCard(QGroupBox):
    def __init__(self, title: str, metric_name: str = "Score"):
        super().__init__(title)

        self.metric_name_label = QLabel(metric_name)
        self.metric_name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.score_label = QLabel("-")
        self.score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.score_label.setFont(QFont("Arial", 22, QFont.Weight.Bold))

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("%p%")

        self.details_box = QTextEdit()
        self.details_box.setReadOnly(True)
        self.details_box.setMinimumHeight(180)

        layout = QVBoxLayout()
        layout.addWidget(self.metric_name_label)
        layout.addWidget(self.score_label)
        layout.addWidget(self.progress)
        layout.addWidget(self.details_box)
        self.setLayout(layout)

    def set_score(self, score: float | None):
        if score is None:
            self.score_label.setText("-")
            self.progress.setValue(0)
            return
        self.score_label.setText(f"{score:.4f}")
        self.progress.setValue(score_to_percent(score))

    def set_details(self, details: dict | None):
        if not details:
            self.details_box.setPlainText("No details available.")
            return
        self.details_box.setPlainText(json.dumps(details, indent=2))

    def clear(self):
        self.score_label.setText("-")
        self.progress.setValue(0)
        self.details_box.clear()


class StatusBanner(QLabel):
    def __init__(self):
        super().__init__("Ready.")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setStyleSheet("""
            QLabel {
                background-color: #eef3f8;
                border: 1px solid #c9d6e2;
                padding: 10px;
                font-weight: 600;
            }
        """)

    def set_info(self, text: str):
        self.setText(text)
        self.setStyleSheet("""
            QLabel {
                background-color: #eef3f8;
                border: 1px solid #c9d6e2;
                padding: 10px;
                font-weight: 600;
            }
        """)

    def set_success(self, text: str):
        self.setText(text)
        self.setStyleSheet("""
            QLabel {
                background-color: #eaf7ea;
                border: 1px solid #9fd49f;
                padding: 10px;
                font-weight: 600;
                color: #146c2e;
            }
        """)

    def set_error(self, text: str):
        self.setText(text)
        self.setStyleSheet("""
            QLabel {
                background-color: #fdecec;
                border: 1px solid #e0a2a2;
                padding: 10px;
                font-weight: 600;
                color: #a12626;
            }
        """)


class FusionSummaryWidget(QGroupBox):
    def __init__(self):
        super().__init__("Fusion Summary")

        self.weighted_label = QLabel("-")
        self.fused_label = QLabel("-")
        self.threshold_label = QLabel("-")
        self.decision_label = QLabel("-")

        for label in [self.weighted_label, self.fused_label, self.threshold_label, self.decision_label]:
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setFont(QFont("Arial", 18, QFont.Weight.Bold))

        grid = QGridLayout()
        grid.addWidget(QLabel("Weighted Score"), 0, 0)
        grid.addWidget(QLabel("Final Fusion Score"), 0, 1)
        grid.addWidget(QLabel("Threshold"), 0, 2)
        grid.addWidget(QLabel("Decision"), 0, 3)

        grid.addWidget(self.weighted_label, 1, 0)
        grid.addWidget(self.fused_label, 1, 1)
        grid.addWidget(self.threshold_label, 1, 2)
        grid.addWidget(self.decision_label, 1, 3)

        self.decision_bar = QProgressBar()
        self.decision_bar.setRange(0, 100)
        self.decision_bar.setValue(0)
        self.decision_bar.setFormat("%p%")

        layout = QVBoxLayout()
        layout.addLayout(grid)
        layout.addWidget(self.decision_bar)
        self.setLayout(layout)

    def set_result(self, weighted: float | None, fused: float | None, threshold: float | None, decision: str | None):
        self.weighted_label.setText("-" if weighted is None else f"{weighted:.4f}")
        self.fused_label.setText("-" if fused is None else f"{fused:.4f}")
        self.threshold_label.setText("-" if threshold is None else f"{threshold:.4f}")
        self.decision_label.setText("-" if decision is None else decision.upper())

        self.decision_bar.setValue(score_to_percent(fused))

        if decision == "accept":
            self.decision_label.setStyleSheet("color: #15803d;")
        elif decision == "reject":
            self.decision_label.setStyleSheet("color: #b91c1c;")
        else:
            self.decision_label.setStyleSheet("")

    def clear(self):
        self.set_result(None, None, None, None)


class FaceTab(QWidget):
    def __init__(self):
        super().__init__()
        self.input_card = ImageInputCard("Face Input", "Select face image...")
        self.metric_card = MetricCard("Face Recognition Output", "Face Similarity Score")

        layout = QVBoxLayout()
        layout.addWidget(self.input_card)
        layout.addWidget(self.metric_card)
        self.setLayout(layout)


class FingerprintTab(QWidget):
    def __init__(self):
        super().__init__()
        self.input_card = ImageInputCard("Fingerprint Input", "Select fingerprint image...")
        self.metric_card = MetricCard("Fingerprint Recognition Output", "Fingerprint Match Score")

        layout = QVBoxLayout()
        layout.addWidget(self.input_card)
        layout.addWidget(self.metric_card)
        self.setLayout(layout)


class PalmTab(QWidget):
    def __init__(self):
        super().__init__()
        self.input_card = ImageInputCard("Palm Input", "Select palm image...")
        self.metric_card = MetricCard("Palm Recognition Output", "Palm Match Score")

        layout = QVBoxLayout()
        layout.addWidget(self.input_card)
        layout.addWidget(self.metric_card)
        self.setLayout(layout)


class FusionTab(QWidget):
    def __init__(self):
        super().__init__()
        self.summary = FusionSummaryWidget()

        self.overview_box = QTextEdit()
        self.overview_box.setReadOnly(True)
        self.overview_box.setMinimumHeight(280)

        group = QGroupBox("Fusion Interpretation")
        group_layout = QVBoxLayout()
        group_layout.addWidget(self.overview_box)
        group.setLayout(group_layout)

        layout = QVBoxLayout()
        layout.addWidget(self.summary)
        layout.addWidget(group)
        self.setLayout(layout)

    def set_overview(self, text: str):
        self.overview_box.setPlainText(text)

    def clear(self):
        self.summary.clear()
        self.overview_box.clear()


class RawResponseTab(QWidget):
    def __init__(self):
        super().__init__()
        self.output = QTextEdit()
        self.output.setReadOnly(True)

        layout = QVBoxLayout()
        layout.addWidget(self.output)
        self.setLayout(layout)

    def set_text(self, text: str):
        self.output.setPlainText(text)

    def clear(self):
        self.output.clear()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multimodal Biometric Authentication System")
        self.resize(1350, 900)

        self.subject_input = QLineEdit()
        self.subject_input.setPlaceholderText("Enter enrolled subject ID, e.g. subject_001")

        self.verify_btn = QPushButton("Run Verification")
        self.clear_btn = QPushButton("Clear")
        self.status_banner = StatusBanner()

        self.tabs = QTabWidget()
        self.face_tab = FaceTab()
        self.fp_tab = FingerprintTab()
        self.palm_tab = PalmTab()
        self.fusion_tab = FusionTab()
        self.raw_tab = RawResponseTab()

        self.tabs.addTab(self.face_tab, "Face")
        self.tabs.addTab(self.fp_tab, "Fingerprint")
        self.tabs.addTab(self.palm_tab, "Palm")
        self.tabs.addTab(self.fusion_tab, "Fusion")
        self.tabs.addTab(self.raw_tab, "Raw Response")

        self.verify_btn.clicked.connect(self.verify)
        self.clear_btn.clicked.connect(self.clear_all)

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        title = QLabel("Multimodal Authentication Dashboard")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Arial", 20, QFont.Weight.Bold))

        subtitle = QLabel(
            "Demonstration of Face, Fingerprint, Palm, and Fusion-based Verification"
        )
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #555;")

        control_box = QGroupBox("Verification Controls")
        control_layout = QHBoxLayout()
        control_layout.addWidget(QLabel("Subject ID:"))
        control_layout.addWidget(self.subject_input)
        control_layout.addWidget(self.verify_btn)
        control_layout.addWidget(self.clear_btn)
        control_box.setLayout(control_layout)

        layout = QVBoxLayout()
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(control_box)
        layout.addWidget(self.status_banner)
        layout.addWidget(self.tabs)

        central.setLayout(layout)

    def clear_all(self):
        self.subject_input.clear()
        self.face_tab.input_card.clear()
        self.fp_tab.input_card.clear()
        self.palm_tab.input_card.clear()

        self.face_tab.metric_card.clear()
        self.fp_tab.metric_card.clear()
        self.palm_tab.metric_card.clear()

        self.fusion_tab.clear()
        self.raw_tab.clear()

        self.status_banner.set_info("Cleared.")

    def _validate_inputs(self) -> bool:
        subject_id = self.subject_input.text().strip()
        if not subject_id:
            QMessageBox.warning(self, "Missing Subject ID", "Please enter a subject ID.")
            return False

        paths = [
            self.face_tab.input_card.get_path(),
            self.fp_tab.input_card.get_path(),
            self.palm_tab.input_card.get_path(),
        ]

        if not any(paths):
            QMessageBox.warning(
                self,
                "Missing Inputs",
                "Please select at least one image for face, fingerprint, or palm."
            )
            return False

        for path in paths:
            if path and not Path(path).exists():
                QMessageBox.warning(self, "Invalid File", f"File does not exist:\n{path}")
                return False

        return True

    def verify(self):
        if not self._validate_inputs():
            return

        payload = {
            "subject_id": self.subject_input.text().strip(),
            "face_path": self.face_tab.input_card.get_path(),
            "fingerprint_path": self.fp_tab.input_card.get_path(),
            "palm_path": self.palm_tab.input_card.get_path(),
        }

        self.status_banner.set_info("Sending request to backend...")
        QApplication.processEvents()

        try:
            resp = requests.post(API_URL, json=payload, timeout=180)
            resp.raise_for_status()
            data = resp.json()
            self.populate_results(data)
            self.status_banner.set_success("Verification completed successfully.")
        except requests.exceptions.RequestException as e:
            self.status_banner.set_error(f"Request failed: {e}")
            self.raw_tab.set_text(str(e))
            QMessageBox.critical(self, "Request Failed", str(e))
        except ValueError as e:
            self.status_banner.set_error(f"Invalid JSON response: {e}")
            self.raw_tab.set_text(str(e))
            QMessageBox.critical(self, "Invalid Response", str(e))

    def populate_results(self, data: dict):
        self.raw_tab.set_text(json.dumps(data, indent=2))

        modality_scores = data.get("modality_scores", {})
        face_score = modality_scores.get("face")
        fp_score = modality_scores.get("fingerprint")
        palm_score = modality_scores.get("palm")

        fingerprint_details = modality_scores.get("fingerprint_details")
        palm_details = modality_scores.get("palm_details")

        face_details = {
            "score_type": "embedding similarity",
            "face_score": face_score,
            "note": "Face module output based on the configured face embedding pipeline."
        } if face_score is not None else None

        self.face_tab.metric_card.set_score(face_score)
        self.face_tab.metric_card.set_details(face_details)

        self.fp_tab.metric_card.set_score(fp_score)
        self.fp_tab.metric_card.set_details(fingerprint_details)

        self.palm_tab.metric_card.set_score(palm_score)
        self.palm_tab.metric_card.set_details(palm_details)

        weighted = data.get("weighted_score")
        fused = data.get("fused_score")
        threshold = data.get("threshold")
        decision = data.get("decision")

        self.fusion_tab.summary.set_result(weighted, fused, threshold, decision)
        self.fusion_tab.set_overview(self.build_fusion_explanation(data))

    def build_fusion_explanation(self, data: dict) -> str:
        modality_scores = data.get("modality_scores", {})
        face_score = modality_scores.get("face")
        fp_score = modality_scores.get("fingerprint")
        palm_score = modality_scores.get("palm")
        weighted = data.get("weighted_score")
        fused = data.get("fused_score")
        threshold = data.get("threshold")
        decision = data.get("decision")

        lines = [
            "Fusion Decision Explanation",
            "==========================",
            "",
            f"Face score: {face_score if face_score is not None else 'N/A'}",
            f"Fingerprint score: {fp_score if fp_score is not None else 'N/A'}",
            f"Palm score: {palm_score if palm_score is not None else 'N/A'}",
            "",
            f"Weighted fusion score: {weighted if weighted is not None else 'N/A'}",
            f"Final fusion model score: {fused if fused is not None else 'N/A'}",
            f"Decision threshold: {threshold if threshold is not None else 'N/A'}",
            f"Final decision: {decision.upper() if decision else 'N/A'}",
            "",
            "Interpretation:",
        ]

        if fused is not None and threshold is not None and decision is not None:
            if decision == "accept":
                lines.append(
                    f"- The fused score ({fused:.4f}) is above the threshold ({threshold:.4f}), so the identity claim is accepted."
                )
            else:
                lines.append(
                    f"- The fused score ({fused:.4f}) is below the threshold ({threshold:.4f}), so the identity claim is rejected."
                )

        if face_score is not None:
            lines.append(f"- Face module contributed a score of {face_score:.4f}.")
        if fp_score is not None:
            lines.append(f"- Fingerprint module contributed a score of {fp_score:.4f}.")
        if palm_score is not None:
            lines.append(f"- Palm module contributed a score of {palm_score:.4f}.")

        lines.append("")
        lines.append("This tab is intended for demo and presentation use, to explain how individual modality outputs contribute to the final multimodal decision.")

        return "\n".join(lines)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()