from workers.BaseTrainWorker import BaseTrainWorker
from models.model2 import train_classification_model  # замість train_regression_model

class ClassificationTrainWorker(BaseTrainWorker):
    """Worker for training classification models."""

    def run(self):
        try:
            model, acc, history, encoder, scaler, X = train_classification_model(
                file=self.file_path,
                target_column=self.target_column,
                epochs=self.epochs,
                layers=self.layers,
                dropout=self.dropout,
                progress_callback=self.progress,
                stop_callback=self.stop_requested
            )

            if not self.isInterruptionRequested():
                self.finished.emit(model, acc, history, encoder, X)

        except Exception as e:
            # При помилці UI не показує "навчання"
            self.message.emit(f"Classification training error: {str(e)}")
            self.finished.emit(None, 0.0, None, None, None)
