import sys 
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton
from PyQt5.QtGui import QFont, QPixmap

from AkStock_2 import AkStock


class StockWindows(QWidget):
    def __init__(self):
        super().__init__() # create default constructor for QWidget
        self.initializeUI()

    def initializeUI(self):
        """
        Initialize the window and display its contents to the screen. 
        """
        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle('Tao Zhang\'s Algorithmic Trading Platform 1.0')
        self.displayButton()

        self.show()

    def displayButton(self):
        '''
        Setup the button widget. 
        '''
            
        button = QPushButton('Renew', self)
        button.setFont(QFont('Arial', 10))
        button.clicked.connect(self.buttonClicked)
        button.move(50, 560)

        button_2 = QPushButton('Show', self)
        button_2.setFont(QFont('Arial', 10))
        button_2.clicked.connect(self.buttonClicked)
        button_2.move(150, 560)

    def buttonClicked(self):
        '''
        Print message to the terminal, 
        and close the window when button is clicked.
        '''
        print("Get Stock data.")
        s = AkStock()
        s.show()

        print('Done')

 
        



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = StockWindows()
    sys.exit(app.exec_())
