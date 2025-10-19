
from workers.BaseTrainWorker import BaseTrainWorker
from models.model1 import train_regression_model

class RegressionTrainWorker(BaseTrainWorker):
    """Worker for training regression models."""

    def run(self):
        try:
            model, mse, history, scaler, X = train_regression_model(
                file=self.file_path,
                target_column=self.target_column,
                epochs=self.epochs,
                layers=self.layers,
                dropout=self.dropout,
                progress_callback=self.progress,
                stop_callback=self.stop_requested
            )

            if not self.isInterruptionRequested():
                # encoder = None for regression
                model.save("temp_model.keras")
                self.finished.emit(None, float(mse), None, None, list(X))


        except Exception as e:
            self.message.emit(f"❌ Regression training error: {str(e)}")
