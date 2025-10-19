from PyQt5.QtWidgets import QMainWindow
from ui.mainWindow_ui import Ui_MainWindow
from ui.model1_page import Model1Page
from ui.model2_page import Model2Page

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.mini_menu_widget.hide()
        self.ui.content.setCurrentIndex(0)
        self.ui.model_1_btn_2.setChecked(True)

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

        self.model1_tab = Model1Page(self.ui)
        self.model2_tab = Model2Page(self.ui)
