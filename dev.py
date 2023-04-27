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

class VideoWorker(QThread):
    changePixmap = pyqtSignal(QImage)

    def __init__(self, gui):
        super().__init__(gui)
        self.gui = gui
        self.stop_signal = False

    @torch.no_grad()
    def run(self):
        weights="yolov7-w6-pose.pt"
        device = select_device("0") #select device

        strip_optimizer("0", "yolov7-w6-pose.pt")

        model = attempt_load(weights, map_location=device)  #Load model
        model.eval()
        names = model.module.names if hasattr(model, "module") else model.names  # get class names
        cap = cv2.VideoCapture(0)    #pass video to videocapture object
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.gui.cam_width)
        while True:
            ret, frame = cap.read()  #get frame and success from video capture
            if ret is False: #if success is true, means frame exist
                break
            dst = cv2.resize(frame, dsize=(self.gui.cam_width, self.gui.cam_height), interpolation=cv2.INTER_AREA)
            inp = cv2.resize(frame, (640, int((640*self.gui.cam_height)/self.gui.cam_width)))
            inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB) #convert frame to RGB
            inp = letterbox(inp, (self.gui.cam_width), stride=64, auto=True)[0]
            inp = transforms.ToTensor()(inp)
            inp = torch.tensor(np.array([inp.numpy()]))
            inp = inp.to(device)  #convert inp data to device
            inp = inp.float() #convert inp to float precision (cpu)
            with torch.no_grad():  #get predictions
                output_data, _ = model(inp)
            output_data = non_max_suppression_kpt(
                            output_data,
                            0.25,   # Conf. Threshold.
                            0.65, # IoU Threshold.
                            nc=model.yaml["nc"], # Number of classes.
                            nkpt=model.yaml["nkpt"], # Number of keypoints.
                            kpt_label=True)
        
            # for pose in output_data:  # detections per img
            #     for det_index, (*xyxy, conf, cls) in enumerate(reversed(pose[:,:6])): #loop over poses for drawing on frame
            #         c = int(cls)  # integer class
            #         kpts = pose[det_index, 6:]
            #         label = f"{names[c]} {conf:.2f}"
            #         plot_one_box_kpt(xyxy, dst, label=label, color=colors(c, True), 
            #                     line_thickness=6, kpt_label=True, kpts=kpts, steps=3, 
            #                     orig_shape=dst.shape[:2])
            for det in output_data[0]:
                if det[4] < 0.5:
                    continue
                for i in range(len(det)):
                    if i in [6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54]:
                        dst = cv2.circle(dst, (int(det[i]), int(det[i+1])), 3, (0, 255, 0), -1, cv2.LINE_AA)
                        dst = cv2.putText(dst, "{:.2f}".format(det[i+2].item()), (int(det[i]+5), int(det[i+1]-4)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

            rgbImage = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
            convertToQtFormat = QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QImage.Format_RGB888)
            p = convertToQtFormat.scaled(self.gui.cam_width, self.gui.cam_height, Qt.KeepAspectRatio)
            self.changePixmap.emit(p)
                    
            if self.stop_signal:
                self.gui.label.setText("no image")
                break
        cap.release()

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.width = 1600
        self.height = 900
        self.cam_width = 1280
        self.cam_height = 720
        self.start()

    def start(self):
        self.setWindowTitle("ANGEL X")
        self.move(0, 0)
        self.resize(self.width, self.height)

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
        self.label.resize(self.cam_width, self.cam_height)
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