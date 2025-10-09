from PyQt5.QtWidgets import QMainWindow
from ui.mainWindow_ui import Ui_MainWindow
from ui.model1_page import Model1Page


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Початкові налаштування
        self.ui.mini_menu_widget.hide()
        self.ui.content.setCurrentIndex(0)
        self.ui.model_1_btn_2.setChecked(True)

        # Прив’язка кнопок для перемикання сторінок
        self.ui.model_1_btn_1.clicked.connect(lambda: self.ui.content.setCurrentIndex(0))
        self.ui.model_1_btn_2.clicked.connect(lambda: self.ui.content.setCurrentIndex(0))
        self.ui.model_2_btn_1.clicked.connect(lambda: self.ui.content.setCurrentIndex(1))
        self.ui.model_2_btn_2.clicked.connect(lambda: self.ui.content.setCurrentIndex(1))
        self.ui.model_3_btn_1.clicked.connect(lambda: self.ui.content.setCurrentIndex(2))
        self.ui.model_3_btn_2.clicked.connect(lambda: self.ui.content.setCurrentIndex(2))
        self.ui.model_4_btn_1.clicked.connect(lambda: self.ui.content.setCurrentIndex(3))
        self.ui.model_4_btn_2.clicked.connect(lambda: self.ui.content.setCurrentIndex(3))
        self.ui.model_5_btn_1.clicked.connect(lambda: self.ui.content.setCurrentIndex(4))
        self.ui.model_5_btn_2.clicked.connect(lambda: self.ui.content.setCurrentIndex(4))

        # Підключення логіки для моделі 1
        self.model1_tab = Model1Page(self.ui)
