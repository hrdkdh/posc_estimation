import sys
import cv2
import torch
import numpy as np
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt, strip_optimizer
from utils.plots import colors, plot_one_box_kpt

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import uic

class VideoWorker(QThread):
    changePixmap = pyqtSignal(QImage)

    def __init__(self, gui):
        super().__init__(gui)
        self.gui = gui
        self.stop_signal = False

    @torch.no_grad()
    def run(self):
        weights="yolov7-w6-pose.pt"
        device = select_device("cpu") #select device

        strip_optimizer("cpu", "yolov7-w6-pose.pt")

        model = attempt_load(weights, map_location=device)  #Load model
        _ = model.eval()
        names = model.module.names if hasattr(model, "module") else model.names  # get class names
        cap = cv2.VideoCapture(0)    #pass video to videocapture object
        frame_width = int(cap.get(3))  #get video frame width
        while True:
            ret, frame = cap.read()  #get frame and success from video capture
            if ret is False: #if success is true, means frame exist
                break
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #convert frame to RGB
            img = letterbox(img, (frame_width), stride=64, auto=True)[0]
            img = transforms.ToTensor()(img)
            img = torch.tensor(np.array([img.numpy()]))
            img = img.to(device)  #convert img data to device
            img = img.float() #convert img to float precision (cpu)
            with torch.no_grad():  #get predictions
                output_data, _ = model(img)
            output_data = non_max_suppression_kpt(
                            output_data,
                            0.25,   # Conf. Threshold.
                            0.65, # IoU Threshold.
                            nc=model.yaml["nc"], # Number of classes.
                            nkpt=model.yaml["nkpt"], # Number of keypoints.
                            kpt_label=True)
        
            im0 = img[0].permute(1, 2, 0) * 255 # Change format [b, c, h, w] to [h, w, c] for displaying the img.
            im0 = im0.cpu().numpy().astype(np.uint8)
            
            im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR) #reshape img format to (BGR)

            for pose in output_data:  # detections per img
                for det_index, (*xyxy, conf, cls) in enumerate(reversed(pose[:,:6])): #loop over poses for drawing on frame
                    c = int(cls)  # integer class
                    kpts = pose[det_index, 6:]
                    label = f"{names[c]} {conf:.2f}"
                    plot_one_box_kpt(xyxy, im0, label=label, color=colors(c, True), 
                                line_thickness=3, kpt_label=True, kpts=kpts, steps=3, 
                                orig_shape=im0.shape[:2])

            rgbImage = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
            convertToQtFormat = QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QImage.Format_RGB888)
            p = convertToQtFormat.scaled(400, 225, Qt.KeepAspectRatio)
            self.changePixmap.emit(p)
                    
            if self.stop_signal:
                self.gui.label.setText("no image")
                break
        cap.release()

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.width = 400
        self.height = 300
        self.start()

    def start(self):
        self.setWindowTitle("ANGEL X")
        self.move(0, 0)
        self.resize(800, 600)

        self.cb = QComboBox(self)
        self.cb.addItem("구동기 모드 선택")
        self.cb.addItem("Squat")
        self.cb.addItem("Stoop")
        self.cb.addItem("Heavy Load")
        self.cb.activated[str].connect(self.onSelectChanged)
        self.cb.setStyleSheet("QComboBox { max-width:200px; }")

        btn_rec = QPushButton(self)
        btn_rec.setText("REC")
        btn_rec.setStyleSheet("QPushButton { max-width:100px; }")
        self.btn_rec_state = False

        grid = QGridLayout()
        self.setLayout(grid)
        grid.addWidget(self.cb, 0, 0)
        grid.addWidget(btn_rec, 0, 1)
        btn_rec.clicked.connect(self.onRecClicked)

        self.label = QLabel(self)
        self.label.setText("no image")
        self.label.resize(400, 225)
        grid.addWidget(self.label, 1, 0, 1, 2)

        self.center()
        self.show()

    def onSelectChanged(self, text):
        pass

    def onRecClicked(self):
        if self.cb.currentText() == "구동기 모드 선택":
            QMessageBox.about(self, "Error", "구동기 모드를 선택해주세요.")
            return
        
        if self.btn_rec_state == True:
            self.btn_rec_state = False
            self.video_worker.stop_signal = True
            self.video_worker.quit()
            self.video_worker.wait(500)
            return

        self.btn_rec_state = True
        self.video_worker = VideoWorker(self)
        self.video_worker.changePixmap.connect(self.showImage)
        try:
            self.video_worker.start()
        except:
            self.video_worker.quit()
            self.video_worker.wait(500)

    def showImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))


    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())