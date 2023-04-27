import cv2
import torch
import numpy as np
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt, strip_optimizer
from utils.plots import colors, plot_one_box_kpt

@torch.no_grad()
def pose_estimation():
    weights="yolov7-w6-pose.pt"
    device = select_device("cpu") #select device

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
        cv2.imshow("dst", im0)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    strip_optimizer("cpu", "yolov7-w6-pose.pt")
    run()