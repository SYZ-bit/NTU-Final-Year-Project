from __future__ import annotations

import sys
from PyQt6.QtWidgets import QApplication
from frontend.common_ui import SingleModalityWindow


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = SingleModalityWindow(
        title="Fingerprint Verification",
        endpoint="http://127.0.0.1:5000/verify/fingerprint",
        enrollment_label="Enrollment fingerprint",
        probe_label="Probe fingerprint",
    )
    win.resize(700, 750)
    win.show()
    sys.exit(app.exec())