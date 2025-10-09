from PyQt5.QtCore import QThread, pyqtSignal
from models.model1 import train_iris_model  # якщо в тебе є така функція

class TrainWorker(QThread):
    finished = pyqtSignal(object, float, object, object, object)
    progress = pyqtSignal(int)

    def __init__(self, file_path, epochs, layers, dropout):
        super().__init__()
        self.file_path = file_path
        self.epochs = epochs
        self.layers = layers
        self.dropout = dropout

    def run(self):
        def stop_requested():
            return self.isInterruptionRequested()

        model, acc, history, encoder, X = train_iris_model(
            file=self.file_path,
            epochs=self.epochs,
            layers=self.layers,
            dropout=self.dropout,
            progress_callback=self.progress,
            stop_callback=stop_requested
        )
        # якщо користувач натиснув Reset — просто не емiтимо finished
        if not self.isInterruptionRequested():
            self.finished.emit(model, acc, history, encoder, X)
