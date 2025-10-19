import tensorflow as tf
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QWidget, QFileDialog, QVBoxLayout, QPushButton, QLabel, QMessageBox, QComboBox
from workers.RegressionTrainWorker import RegressionTrainWorker
from ui.layer_widget import LayerWidget
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class Model1Page(QWidget):
    def __init__(self, ui):
        super().__init__()
        self.ui = ui

        self.selected_file = None
        self.training_thread = None
        self.trained_model = None
        self.scaler = None
        self.X_data = None
        self.encoder = None  # для категоріальних колонок
        self.layer_widgets = []

        # Container for layers
        self.layers_layout = QVBoxLayout()
        self.ui.model_1_layers_container.setLayout(self.layers_layout)

        # Add layer button
        self.add_layer_btn = QPushButton("Add layer")
        self.add_layer_btn.clicked.connect(self.add_layer)
        self.layers_layout.addWidget(self.add_layer_btn)

        # Control buttons
        self.ui.model_1_start_learning.clicked.connect(self.start_learning)
        self.ui.model_1_upload_data.clicked.connect(self.select_file)
        self.ui.model_1_save.clicked.connect(self.save_model)
        self.ui.model_1_upload_model.clicked.connect(self.load_model)
        self.ui.model_1_reset.clicked.connect(self.reset_model)

        self.ui.model_1_save.setEnabled(False)
        self.ui.model_1_progress.setValue(0)
        self.ui.model_1_progress.setFormat("Learning: %p%")
        self.ui.model_1_progress.hide()

        # Initial layer
        self.add_layer()

    def add_layer(self):
        widget = LayerWidget(len(self.layer_widgets), self.remove_layer)
        self.layer_widgets.append(widget)
        self.layers_layout.insertWidget(len(self.layer_widgets) - 1, widget)

    def remove_layer(self, widget):
        self.layer_widgets.remove(widget)
        widget.setParent(None)
        for i, w in enumerate(self.layer_widgets):
            w.label.setText(f"Layer {i + 1}:")

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Excel file for training",
            "",
            "Excel Files (*.xls *.xlsx *.csv);;All Files (*)"
        )
        if not file_path:
            return

        self.selected_file = file_path
        self.ui.model_1_file.setText(f"Selected file: {file_path.split('/')[-1]}")

        try:
            df = pd.read_excel(file_path, nrows=0) if file_path.endswith((".xls", ".xlsx")) else pd.read_csv(file_path, nrows=0)
            columns = df.columns.tolist()

            self.ui.model_1_target_column.clear()
            self.ui.model_1_target_column.addItems(columns)
            self.ui.model_1_target_column.setEnabled(True)
            self.ui.model_1_target_column.show()
        except Exception as e:
            self.show_error(f"Error reading file: {e}")

    def start_learning(self):
        if not self.selected_file:
            self.show_error("Please select a file first.")
            return

        target_column = self.ui.model_1_target_column.currentText()
        if not target_column:
            self.show_error("Please select a target column.")
            return

        try:
            epochs = int(self.ui.model_1_epochs_input.text() or "15")
            dropout = float(self.ui.model_1_dropout_input.text() or "0.2")
        except ValueError:
            self.show_error("Error: invalid numeric value.")
            return

        layers_config = [(w.neurons.value(), w.activation.currentText()) for w in self.layer_widgets]
        if not layers_config:
            self.show_error("Error: add at least one layer.")
            return

        # Починаємо UI стан навчання
        self.ui.model_1_progress.show()
        self.ui.model_1_progress.setValue(0)
        self.ui.model_1_start_learning.setEnabled(False)
        self.ui.model_1_start_learning.setText("Training...")

        try:
            self.training_thread = RegressionTrainWorker(
                file_path=self.selected_file,
                target_column=target_column,
                epochs=epochs,
                layers=layers_config,
                dropout=dropout
            )
            self.training_thread.progress.connect(self.ui.model_1_progress.setValue)
            self.training_thread.message.connect(self.show_error)
            self.training_thread.finished.connect(self.training_done)
            self.training_thread.start()
        except Exception as e:
            self.show_error(f"Training start error: {e}")

    def training_done(self, model, loss, history, scaler, X_data, encoder=None):
        self.ui.model_1_progress.hide()
        self.trained_model = tf.keras.models.load_model("temp_model.keras")

        self.scaler = scaler
        self.X_data = X_data
        self.encoder = encoder

        self.ui.model_1_save.setEnabled(True)
        self.ui.model_1_start_learning.setEnabled(True)
        self.ui.model_1_start_learning.setText("Start learning")
        self.ui.model_1_container.setCurrentIndex(1)

        self.show_predictions(loss, history)

    def show_predictions(self, loss=None, history=None):
        if not self.trained_model or self.X_data is None:
            self.show_error("Model is not trained or data is missing.")
            return

        X_input = self.X_data.copy()
        if self.encoder:
            X_input = self.encoder.transform(X_input)

        preds = self.trained_model.predict(X_input).flatten()
        n = self.ui.model_1_prediction_count.value()
        n = min(n, len(preds))
        sample_predictions = "\n".join([f"{i + 1}: {val:.4f}" for i, val in enumerate(preds[:n])])

        info = ""
        if loss is not None and history is not None:
            info += (
                f"Model trained.\n"
                f"Test loss: {loss:.4f}\n"
                f"Epochs: {len(history.history['loss'])}\n"
                f"Number of layers: {len(self.layer_widgets)}\n\n"
            )

        info += f"Predictions (first {n} rows):\n{sample_predictions}"
        self.ui.model_1_result.setText(info)

    def save_model(self):
        if not self.trained_model:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save model as",
            "",
            "H5 Files (*.h5)"
        )
        if file_path:
            if not file_path.endswith(".h5"):
                file_path += ".h5"
            self.trained_model.save(file_path)
            self.ui.model_1_result.setText(f"Model saved at: {file_path}")

    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select model to load",
            "",
            "H5 Files (*.h5);;All Files (*)"
        )
        if file_path:
            self.trained_model = tf.keras.models.load_model(file_path)
            self.ui.model_1_result.setText(f"Model loaded: {file_path}")
            self.ui.model_1_container.setCurrentIndex(1)
            self.ui.model_1_save.setEnabled(True)
            self.ui.model_1_upload_model.setEnabled(True)
            if self.X_data is not None:
                self.show_predictions()

    def reset_model(self):
        if self.training_thread:
            if self.training_thread.isRunning():
                self.training_thread.requestInterruption()
            self.training_thread = None

        self.trained_model = None
        self.scaler = None
        self.X_data = None
        self.encoder = None
        self.selected_file = None

        self.ui.model_1_result.clear()
        self.ui.model_1_file.setText("")
        self.ui.model_1_save.setEnabled(False)
        self.ui.model_1_start_learning.setEnabled(True)
        self.ui.model_1_start_learning.setText("Start learning")
        self.ui.model_1_container.setCurrentIndex(0)

        for w in self.layer_widgets:
            w.setParent(None)
        self.layer_widgets.clear()
        self.add_layer()

        self.ui.model_1_progress.setValue(0)
        self.ui.model_1_progress.hide()

        self.ui.model_1_epochs_input.clear()
        self.ui.model_1_dropout_input.clear()
        self.ui.model_1_prediction_count.setValue(40)
        self.ui.model_1_target_column.hide()
        self.ui.model_1_target_column.clear()

    def show_error(self, message):
        self.reset_model()
        QMessageBox.critical(self, "Error", message)
