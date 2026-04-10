from __future__ import annotations

import sys
from PyQt6.QtWidgets import QApplication
from frontend.common_ui import SingleModalityWindow


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = SingleModalityWindow(
        title="Palm Verification",
        endpoint="http://127.0.0.1:5000/verify/palm",
        enrollment_label="Enrollment palm",
        probe_label="Probe palm",
    )
    win.resize(700, 750)
    win.show()
    sys.exit(app.exec())