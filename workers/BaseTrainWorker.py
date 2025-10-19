from PyQt5.QtCore import QThread, pyqtSignal

class BaseTrainWorker(QThread):
    """Abstract base class for training workers."""

    finished = pyqtSignal(object, float, object, object, object)
    progress = pyqtSignal(int)
    message = pyqtSignal(str)

    def __init__(self, file_path, target_column,
                 epochs=30, layers=None, dropout=0.2):
        super().__init__()
        self.file_path = file_path
        self.target_column = target_column
        self.epochs = epochs
        self.layers = layers
        self.dropout = dropout

    def stop_requested(self):
        """Check if training should stop."""
        return self.isInterruptionRequested()

    def run(self):
        """Template method — to be overridden in subclasses."""
        raise NotImplementedError("Subclasses must implement this method.")
