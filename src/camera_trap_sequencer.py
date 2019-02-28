import sys
import os

from PySide2 import QtCore, QtWidgets, QtGui


class CameraTrapSequencer(QtWidgets.QWidget):
    """
    A simple GUI for ordering databases and directories of camera trap images
    into sequences based on the camera that was used to take the picture and
    the time of creation.

    :author: Joschka Strüber
    """
    def __init__(self):
        super().__init__()

        # create all widgets
        self.directory_type = QtWidgets.QComboBox()
        self.directory_type.addItem("Database")
        self.directory_type.addItem("Directory")

        self.move_method = QtWidgets.QComboBox()
        self.move_method.addItem("Copy")
        self.move_method.addItem("Move")

        self.empty_info = QtWidgets.QCheckBox("Empty Images")
        self.empty_info.setChecked(True)

        self.input_button = QtWidgets.QPushButton("Choose Input Directory")
        self.input_button.clicked.connect(self.get_input_dir)

        self.output_button = QtWidgets.QPushButton("Choose Output Directory")
        self.output_button.clicked.connect(self.get_output_dir)

        self.order_button = QtWidgets.QPushButton("Order Sequences")
        self.order_button.clicked.connect(self.order_sequences)

        # todo: add widgets for input and output paths

        # inner class for horizontal lines that can be used for separation
        class VerticalSeparator(QtWidgets.QFrame):
            def __init__(self):
                super().__init__()
                self.setFrameShape(QtWidgets.QFrame.HLine)
                self.setFrameShadow(QtWidgets.QFrame.Sunken)

        # create layout and add widgets
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.directory_type)
        self.layout.addWidget(VerticalSeparator())
        self.layout.addWidget(self.move_method)
        self.layout.addWidget(self.empty_info)
        self.layout.addWidget(self.input_button)
        self.layout.addWidget(self.output_button)
        self.layout.addWidget(VerticalSeparator())
        self.layout.addWidget(self.order_button)

        self.setLayout(self.layout)
        self.setWindowTitle("Camera Trap Sequencer")

    def order_sequences(self):
        pass

    def get_input_dir(self):
        self.get_dir("Choose Input Directory")

    def get_output_dir(self):
        self.get_dir("Choose Output Directory")

    def get_dir(self, caption):
        dir_name = QtWidgets.QFileDialog.getExistingDirectory(self, caption,
                                                              os.path.expanduser("~"),
                                                              option=QtWidgets.QFileDialog.ShowDirsOnly
                                                         )


if __name__ == '__main__':
    app = QtWidgets.QApplication([])

    widget = CameraTrapSequencer()
    widget.resize(widget.sizeHint())
    widget.show()

    sys.exit(app.exec_())
