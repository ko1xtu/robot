import sys

import cv2
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from models import AudioPipeline, Detector


class Window(QMainWindow):
    def __init__(self, detector, audio, cam_index=0):
        super().__init__()
        self.detector = detector
        self.audio = audio
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {cam_index}")

        self.video = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.status = QLabel("—")
        self.transcript = QPlainTextEdit(readOnly=True)
        self.transcript.setMaximumHeight(120)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(self.video, stretch=1)
        layout.addWidget(self.transcript)
        self.setCentralWidget(central)
        self.statusBar().addWidget(self.status)

        self.audio.text_ready.connect(self.on_text)
        self.audio.start()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(30)             # ~33 fps

    def on_text(self, translation, label):
        self.transcript.appendPlainText(f"[{label or '-'}] {translation}")

    def tick(self):
        ok, frame = self.cap.read()
        if not ok:
            return

        self.detector.detect(frame)
        annotated = self.detector.get_annotated_frame()
        phone_conf = self.detector.get_phone_confidence()
        if annotated is None:
            return

        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self.video.setPixmap(QPixmap.fromImage(img).scaled(
            self.video.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))

        self.status.setText(
            f"phone: {phone_conf:.2f}" if phone_conf is not None else "phone: none"
        )

    def closeEvent(self, e):
        self.timer.stop()
        self.cap.release()
        self.audio.stop()          # sets flag; thread exits within ~32 ms
        e.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Window(Detector(), AudioPipeline())
    w.resize(900, 800)
    w.show()
    sys.exit(app.exec())
