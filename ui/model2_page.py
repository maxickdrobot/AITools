import tensorflow as tf
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QWidget, QFileDialog, QVBoxLayout, QPushButton, QLabel, QMessageBox
from workers.ClassificationTrainWorker import ClassificationTrainWorker
from ui.layer_widget import LayerWidget

class Model2Page(QWidget):
    def __init__(self, ui):
        super().__init__()
        self.ui = ui

        self.selected_file = None
        self.training_thread = None
        self.trained_model = None
        self.encoder = None
        self.X_data = None
        self.layer_widgets = []

        # Container for layers
        self.layers_layout = QVBoxLayout()
        self.ui.model_2_layers_container.setLayout(self.layers_layout)

        # Add layer button
        self.add_layer_btn = QPushButton("Add layer")
        self.add_layer_btn.clicked.connect(self.add_layer)
        self.layers_layout.addWidget(self.add_layer_btn)

        # Control buttons
        self.ui.model_2_start_learning.clicked.connect(self.start_learning)
        self.ui.model_2_upload_data.clicked.connect(self.select_file)
        self.ui.model_2_save.clicked.connect(self.save_model)
        self.ui.model_2_upload_model.clicked.connect(self.load_model)
        self.ui.model_2_reset.clicked.connect(self.reset_model)

        self.ui.model_2_save.setEnabled(False)
        self.ui.model_2_progress.setValue(0)
        self.ui.model_2_progress.setFormat("Learning: %p%")
        self.ui.model_2_progress.hide()

        self.ui.model_2_target_column.hide()

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
        self.ui.model_2_file.setText(f"Selected file: {file_path.split('/')[-1]}")

        try:
            df = pd.read_excel(file_path, nrows=0) if file_path.endswith((".xls", ".xlsx")) else pd.read_csv(file_path, nrows=0)
            columns = df.columns.tolist()

            self.ui.model_2_target_column.clear()
            self.ui.model_2_target_column.addItems(columns)
            self.ui.model_2_target_column.setEnabled(True)
            self.ui.model_2_target_column.show()
        except Exception as e:
            self.show_error(f"Error reading file: {e}")

    def start_learning(self):
        if not self.selected_file:
            self.show_error("Please select a file first.")
            return

        target_column = self.ui.model_2_target_column.currentText()
        if not target_column:
            self.show_error("Please select a target column.")
            return

        try:
            epochs = int(self.ui.model_2_epochs_input.text() or "30")
            dropout = float(self.ui.model_2_dropout_input.text() or "0.2")
        except ValueError:
            self.show_error("Error: invalid numeric value.")
            return

        layers_config = [(w.neurons.value(), w.activation.currentText()) for w in self.layer_widgets]
        if not layers_config:
            self.show_error("Error: add at least one layer.")
            return

        # Починаємо UI стан навчання
        self.ui.model_2_progress.show()
        self.ui.model_2_progress.setValue(0)
        self.ui.model_2_start_learning.setEnabled(False)
        self.ui.model_2_start_learning.setText("Training...")

        try:
            self.training_thread = ClassificationTrainWorker(
                file_path=self.selected_file,
                target_column=target_column,
                epochs=epochs,
                layers=layers_config,
                dropout=dropout
            )
            self.training_thread.progress.connect(self.ui.model_2_progress.setValue)
            # Замість lambda → викликаємо нашу show_error, яка скине UI
            self.training_thread.message.connect(self.show_error)
            self.training_thread.finished.connect(self.training_done)
            self.training_thread.start()
        except Exception as e:
            # Якщо помилка при створенні воркера/старті навчання
            self.show_error(f"Training start error: {e}")

    def training_done(self, model, acc, history, encoder, X_data):
        self.ui.model_2_progress.hide()
        self.trained_model = model
        self.encoder = encoder
        self.X_data = X_data

        self.ui.model_2_save.setEnabled(True)
        self.ui.model_2_start_learning.setEnabled(True)
        self.ui.model_2_start_learning.setText("Start learning")
        self.ui.model_2_container.setCurrentIndex(1)

        self.show_predictions(acc, history)

    def show_predictions(self, acc=None, history=None):
        if not self.trained_model or self.X_data is None or self.encoder is None:
            self.show_error("Model is not trained or data is missing.")
            return

        logits = self.trained_model.predict(self.X_data)
        pred_indices = np.argmax(logits, axis=1)
        pred_classes = self.encoder.inverse_transform(pred_indices)

        n = self.ui.model_2_prediction_count.value()
        n = min(n, len(pred_classes))
        sample_predictions = "\n".join([f"{i + 1}: {cls}" for i, cls in enumerate(pred_classes[:n])])

        info = ""
        if acc is not None and history is not None:
            info += (
                f"Model trained.\n"
                f"Test accuracy: {acc:.2f}\n"
                f"Epochs: {len(history.history['loss'])}\n"
                f"Number of layers: {len(self.layer_widgets)}\n\n"
            )

        info += f"Predictions (first {n} rows):\n{sample_predictions}"
        self.ui.model_2_result.setText(info)

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
            self.ui.model_2_result.setText(f"Model saved at: {file_path}")

    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select model to load",
            "",
            "H5 Files (*.h5);;All Files (*)"
        )
        if file_path:
            self.trained_model = tf.keras.models.load_model(file_path)
            self.ui.model_2_result.setText(f"Model loaded: {file_path}")
            self.ui.model_2_container.setCurrentIndex(1)
            self.ui.model_2_save.setEnabled(True)
            self.ui.model_2_upload_model.setEnabled(True)
            if self.X_data is not None and self.encoder is not None:
                self.show_predictions()
            elif self.selected_file:
                self.prepare_data_for_prediction(self.selected_file)
            else:
                self.ui.model_2_result.setText(f"Model loaded: {file_path}\nNo data loaded for predictions.")

    def prepare_data_for_prediction(self, file_path):
        from utils.data_loader import load_data_and_encoder
        self.X_data, self.encoder = load_data_and_encoder(file_path)
        self.show_predictions()

    def reset_model(self):
        if self.training_thread:
            if self.training_thread.isRunning():
                self.training_thread.requestInterruption()
            self.training_thread = None

        self.trained_model = None
        self.encoder = None
        self.X_data = None
        self.selected_file = None

        self.ui.model_2_result.clear()
        self.ui.model_2_file.setText("")
        self.ui.model_2_save.setEnabled(False)
        self.ui.model_2_start_learning.setEnabled(True)
        self.ui.model_2_start_learning.setText("Start learning")
        self.ui.model_2_container.setCurrentIndex(0)

        for w in self.layer_widgets:
            w.setParent(None)
        self.layer_widgets.clear()
        self.add_layer()

        self.ui.model_2_progress.setValue(0)
        self.ui.model_2_progress.hide()

        self.ui.model_2_epochs_input.clear()
        self.ui.model_2_dropout_input.clear()
        self.ui.model_2_prediction_count.setValue(40)
        self.ui.model_2_target_column.hide()
        self.ui.model_2_target_column.clear()

    def show_error(self, message):
        self.reset_model()
        QMessageBox.critical(self, "Error", message)