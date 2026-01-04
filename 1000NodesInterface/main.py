import sys
from PyQt6.QtWidgets import QApplication
from ui.gui import MainWindow, apply_modern_theme 

def main():
    print("Large-Scale Network Simulation Starting...")
    try:
        app = QApplication(sys.argv)
        apply_modern_theme(app)
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        print(f"Error launching application: {e}")

if __name__ == "__main__":
    main()
