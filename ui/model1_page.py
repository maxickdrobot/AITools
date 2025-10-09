from PyQt5.QtWidgets import QWidget, QFileDialog, QVBoxLayout, QPushButton, QLabel
from workers.train_worker import TrainWorker
import tensorflow as tf
import numpy as np
from ui.layer_widget import LayerWidget

class Model1Page(QWidget):
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
            "Excel Files (*.xls *.xlsx);;All Files (*)"
        )
        if file_path:
            self.selected_file = file_path
            self.ui.model_1_file.setText(f"Selected file: {file_path.split('/')[-1]}")

    def start_learning(self):
        try:
            epochs = int(self.ui.model_1_epochs_input.text() or "30")
            dropout = float(self.ui.model_1_dropout_input.text() or "0.2")
        except ValueError:
            self.ui.model_1_result.setText("Error: invalid numeric value.")
            return

        layers_config = [(w.neurons.value(), w.activation.currentText()) for w in self.layer_widgets]
        if not layers_config:
            self.ui.model_1_result.setText("Error: add at least one layer.")
            return

        file_to_use = self.selected_file if self.selected_file else "iris.xls"

        self.ui.model_1_progress.show()
        self.ui.model_1_progress.setValue(0)
        self.ui.model_1_start_learning.setEnabled(False)
        self.ui.model_1_start_learning.setText("Training...")

        self.training_thread = TrainWorker(
            file_path=file_to_use,
            epochs=epochs,
            layers=layers_config,
            dropout=dropout
        )
        self.training_thread.progress.connect(self.ui.model_1_progress.setValue)
        self.training_thread.finished.connect(self.training_done)
        self.training_thread.start()

    def training_done(self, model, acc, history, encoder, X_data):
        self.ui.model_1_progress.hide()
        self.trained_model = model
        self.encoder = encoder
        self.X_data = X_data

        self.ui.model_1_save.setEnabled(True)
        self.ui.model_1_start_learning.setEnabled(True)
        self.ui.model_1_start_learning.setText("Start learning")
        self.ui.model_1_container.setCurrentIndex(1)

        self.show_predictions(acc, history)

    def show_predictions(self, acc=None, history=None):
        if not self.trained_model or self.X_data is None or self.encoder is None:
            self.ui.model_1_result.setText("Model is not trained or data is missing.")
            return

        logits = self.trained_model.predict(self.X_data)
        pred_indices = np.argmax(logits, axis=1)
        pred_classes = self.encoder.inverse_transform(pred_indices)

        n = self.ui.model_1_prediction_count.value()
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
            # If data exists, show predictions; else ask to load data
            if self.X_data is not None and self.encoder is not None:
                self.show_predictions()
            elif self.selected_file:
                self.prepare_data_for_prediction(self.selected_file)
            else:
                self.ui.model_1_result.setText(
                    f"Model loaded: {file_path}\nNo data loaded for predictions."
                )

    def prepare_data_for_prediction(self, file_path):
        from utils.data_loader import load_data_and_encoder  # example
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

        # Clear UI
        self.ui.model_1_result.clear()
        self.ui.model_1_file.setText("")
        self.ui.model_1_save.setEnabled(False)
        self.ui.model_1_start_learning.setEnabled(True)
        self.ui.model_1_start_learning.setText("Start learning")
        self.ui.model_1_container.setCurrentIndex(0)

        # Reset layers
        for w in self.layer_widgets:
            w.setParent(None)
        self.layer_widgets.clear()
        self.add_layer()

        # Reset progress
        self.ui.model_1_progress.setValue(0)
        self.ui.model_1_progress.hide()

        # Reset input fields
        self.ui.model_1_epochs_input.clear()
        self.ui.model_1_dropout_input.clear()
        self.ui.model_1_prediction_count.setValue(10)  # default
