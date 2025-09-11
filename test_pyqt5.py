# from PyQt5 import QtWidgets
# from PyQt5.QtWidgets import (
#     QApplication,
#     QMainWindow,
#     QListWidget,
#     QWidget,
#     QVBoxLayout,
#     QLabel,
# )
# import sys


# # drag and drop file
# class DropList(QListWidget):
#     def __init__(self, label):
#         super().__init__()
#         self.setAcceptDrops(True)
#         self.label = label

#     def dragEnterEvent(self, event):
#         if event.mimeData().hasUrls():  # only allow files
#             event.accept()
#         else:
#             event.ignore()

#     def dropEvent(self, event):
#         file_paths = []
#         for url in event.mimeData().urls():
#             file_path = url.toLocalFile()
#             self.addItem(file_path)  # show file in the list
#             file_paths.append(file_path)

#         # update the label
#         self.label.setText("Dropped:\n" + "\n".join(file_paths))


# def window():
#     app = QApplication(
#         sys.argv
#     )  # start off the application, give config setup for QApplication
#     win = QMainWindow()

#     win.setGeometry(
#         200, 200, 300, 300
#     )  # where u want the application to launch at, xpos, ypos, width, height
#     win.setWindowTitle("Steganography with PyQt5")

#     # to put stuff inside the window = = =
#     label = QtWidgets.QLabel(win)  # indicate where the label will show up in
#     label.setText("My First Label")
#     label.move(50, 50)  # move the label, xpos, ypos

#     # show the window
#     win.show()
#     sys.exit(app.exec_())  # let them have a clean exit when app is closed


# def window1():
#     app = QApplication([])

#     window = QWidget()
#     window.setWindowTitle("Drag & Drop Demo")
#     window.resize(500, 300)

#     layout = QVBoxLayout(window)

#     label = QLabel("Drag a file into this window...")
#     drop_list = DropList(label)

#     layout.addWidget(label)
#     layout.addWidget(drop_list)

#     window.setLayout(layout)
#     window.show()
#     app.exec_()

# window1()


from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QListWidget
)
import sys


# Drag and drop list widget
class DropList(QListWidget):
    def __init__(self, label):
        super().__init__()
        self.setAcceptDrops(True)
        self.label = label

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():   # only allow file drops
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            file_paths = []
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                self.addItem(file_path)   # show in list
                file_paths.append(file_path)

            # update label
            self.label.setText("Dropped:\n" + "\n".join(file_paths))
            event.acceptProposedAction()
        else:
            event.ignore()


# Main window
class FileSelectDemo(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("File Select Demo (Drag & Drop + Upload)")
        self.resize(500, 300)

        layout = QVBoxLayout()

        self.label = QLabel("Drag a file here or use the Upload button")
        layout.addWidget(self.label)

        # Drag-and-drop list
        self.drop_list = DropList(self.label)
        layout.addWidget(self.drop_list)

        # Upload button (opens Finder)
        self.upload_button = QPushButton("Upload File")
        self.upload_button.clicked.connect(self.open_file_dialog)
        layout.addWidget(self.upload_button)

        self.setLayout(layout)

    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select a file",
            "",
            "Images (*.png *.bmp);;Audio (*.wav);;All Files (*)"
        )
        if file_path:
            self.drop_list.addItem(file_path)
            self.label.setText(f"Selected: {file_path}")


def main():
    app = QApplication(sys.argv)
    window = FileSelectDemo()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
