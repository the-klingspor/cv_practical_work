import sys

from PySide2 import QtCore, QtWidgets, QtGui


class CameraTrapSequencer(QtWidgets.QWidget):
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

        self.button = QtWidgets.QPushButton("Order Sequences")
        self.button.clicked.connect(self.order_sequences())

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
        self.layout.addWidget(VerticalSeparator())
        self.layout.addWidget(self.button)

        self.setLayout(self.layout)
        self.setWindowTitle("Camera Trap Sequencer")

    def order_sequences(self):
        pass


if __name__ == '__main__':
    app = QtWidgets.QApplication([])

    widget = CameraTrapSequencer()
    widget.resize(widget.sizeHint())
    widget.show()

    sys.exit(app.exec_())
