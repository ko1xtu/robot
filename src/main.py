import sys

from PyQt6.QtWidgets import QApplication

from gui import Window
from models import AudioPipeline, Detector

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Window(Detector(), AudioPipeline())
    w.resize(900, 800)
    w.show()
    sys.exit(app.exec())
