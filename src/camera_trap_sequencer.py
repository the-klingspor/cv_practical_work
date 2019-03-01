import sys
import os

from PyQt5 import QtWidgets, QtCore

from src.sequences import order_db_by_sequences, order_dir_by_sequences


class CameraTrapSequencer(QtWidgets.QWidget):
    """
    A simple GUI for ordering databases and directories of camera trap images
    into sequences based on the camera that was used to take the picture and
    the time of creation.

    :author: Joschka Strüber
    """

    # todo: check for exceptions (exiftool missing)
    # todo: more error handling in case of bad input directories
    # todo: save dirs in case of cancellation
    # todo: set start directory to last selected

    def __init__(self):
        super().__init__()

        self.input_dir = None
        self.output_dir = None
        self.dirs_changed = False

        # create all widgets
        self.directory_type_label = QtWidgets.QLabel("Input type: ")

        self.directory_type = QtWidgets.QComboBox()
        self.directory_type.addItem("Database")
        self.directory_type.addItem("Directory")
        self.directory_type.currentIndexChanged.connect(self._directory_type_change)

        self.move_method_label = QtWidgets.QLabel("Move method: ")

        self.move_method = QtWidgets.QComboBox()
        self.move_method.addItem("Copy")
        self.move_method.addItem("Move")

        self.empty_info_label = QtWidgets.QLabel("Use empty images: ")

        self.empty_info = QtWidgets.QCheckBox("Empty Images")
        self.empty_info.setChecked(True)

        self.input_dir_label = QtWidgets.QLineEdit("Choose input directory")
        self.input_dir_label.setReadOnly(True)
        self.input_dir_label.setFixedWidth(200)

        self.input_button = QtWidgets.QPushButton("Input Directory")
        self.input_button.clicked.connect(self.get_input_dir)

        self.output_dir_label = QtWidgets.QLineEdit("Choose output directory")
        self.output_dir_label.setReadOnly(True)
        self.output_dir_label.setFixedWidth(200)

        self.output_button = QtWidgets.QPushButton("Output Directory")
        self.output_button.clicked.connect(self.get_output_dir)

        self.order_button = QtWidgets.QPushButton("Order Sequences")
        self.order_button.clicked.connect(self.on_start)

        self.progress_bar = QtWidgets.QProgressBar(self)
        self.progress_bar.setRange(0, 1)

        self.sequence_thread = None

        # inner class for horizontal lines that can be used for separation
        class VerticalSeparator(QtWidgets.QFrame):
            def __init__(self):
                super().__init__()
                self.setFrameShape(QtWidgets.QFrame.HLine)
                self.setFrameShadow(QtWidgets.QFrame.Sunken)

        self.layout = QtWidgets.QVBoxLayout()

        # create inner layout and add widgets
        self.inner_layout = QtWidgets.QFormLayout()
        self.inner_layout.addRow(self.directory_type_label, self.directory_type)
        self.inner_layout.addRow(self.move_method_label, self.move_method)
        self.inner_layout.addRow(self.empty_info_label, self.empty_info)
        self.inner_layout.addRow(self.input_dir_label, self.input_button)
        self.inner_layout.addRow(self.output_dir_label, self.output_button)
        self.layout.addLayout(self.inner_layout)
        self.layout.addWidget(VerticalSeparator())
        self.layout.addWidget(self.order_button)
        self.layout.addWidget(self.progress_bar)

        self.setLayout(self.layout)
        self.setWindowTitle("Camera Trap Sequencer")

    def on_start(self):
        if self.input_dir is None:
            no_input_dialog = QtWidgets.QErrorMessage(self)
            no_input_dialog.showMessage("No input directory was chosen.")
        elif self.output_dir is None:
            no_output_dialog = QtWidgets.QErrorMessage(self)
            no_output_dialog.showMessage("No output directory was chosen.")
        elif not self.dirs_changed:
            no_dirs_changed_dialog = QtWidgets.QErrorMessage(self)
            no_dirs_changed_dialog.showMessage("The directories were not changed"
                                               " after the last ordering of sequences.")
        else:
            self.progress_bar.setRange(0, 0)
            self.dirs_changed = False
            move_type = self.directory_type.currentText()
            copy = True if self.move_method.currentText() == "Copy" else False
            empty = self.empty_info.isChecked()
            self.sequence_thread = SequencesThread(move_type=move_type,
                                                   path_from=self.input_dir,
                                                   path_to=self.output_dir,
                                                   copy=copy,
                                                   empty=empty)
            self.sequence_thread.task_finished.connect(self.on_finished)
            self.sequence_thread.start()

    def on_finished(self):
        # Stop the pulsation
        self.progress_bar.setRange(0, 1)

    def get_input_dir(self):
        self.input_dir = self._get_dir("Choose Input Directory")
        self.dirs_changed = True
        self.input_dir_label.setText(self._shorten_dir(self.input_dir))

    def get_output_dir(self):
        self.output_dir = self._get_dir("Choose Output Directory")
        self.dirs_changed = True
        self.output_dir_label.setText(self._shorten_dir(self.output_dir))

    def _get_dir(self, caption):
        dir_name = QtWidgets.QFileDialog.getExistingDirectory(self, caption,
                                                              os.path.expanduser("~"),
                                                              options=QtWidgets.QFileDialog.ShowDirsOnly)
        return dir_name

    def _shorten_dir(self, path):
        parent_dirs, dir_name = os.path.split(path)
        parent_dir = os.path.basename(parent_dirs)
        short_path = os.path.join("...", parent_dir, dir_name)
        return short_path

    def _directory_type_change(self):
        if self.directory_type.currentText() == "Database":
            self.empty_info.setEnabled(True)
        elif self.directory_type.currentText() == "Directory":
            self.empty_info.setEnabled(False)
        else:
            assert False


class SequencesThread(QtCore.QThread):

    def __init__(self, move_type, path_from, path_to, copy, empty, parent=None):
        QtCore.QThread.__init__(self, parent)

        self.move_type = move_type
        self.path_from = path_from
        self.path_to = path_to
        self.copy = copy
        self.empty = empty

    task_finished = QtCore.pyqtSignal()

    def run(self):
        self._order_sequences()
        self.task_finished.emit()

    def _order_sequences(self):
        if self.move_type == "Database":
            order_db_by_sequences(self.path_from,
                                  self.path_to,
                                  self.copy,
                                  self.empty)
        elif self.move_type == "Directory":
            order_dir_by_sequences(self.path_from,
                                   self.path_to,
                                   self.copy)
        else:
            assert False


if __name__ == '__main__':
    app = QtWidgets.QApplication([])

    widget = CameraTrapSequencer()
    widget.resize(widget.sizeHint())
    widget.show()

    sys.exit(app.exec_())
