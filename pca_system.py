import cv2
import numpy as np
import threading
import sys
import os
import time
from pyzbar import pyzbar

from collections import deque
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QObject
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QWidget

sys.path.append('/opt/MVS/Samples/64/Python/MvImport')
from MvCameraControl_class import *

class VideoThread(QObject):
    update_frame = pyqtSignal(QImage)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.buffer = deque()
        self.running = False
        
    def start(self):
        self.running = True
        threading.Thread(target=self.run, daemon=True).start()
    
    def run(self):
        while self.running:
            if self.buffer:
                frame = self.buffer.popleft()
                if frame is not None:
                    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    self.update_frame.emit(qt_image)
                else:
                    print("Failed to capture image")
            time.sleep(0.01)

    def stop(self):
        self.running = False

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Detect Worker')
        
        self.pushButton_open = QtWidgets.QPushButton('Open Camera', self)
        self.pushButton_open.setGeometry(10, 10, 200, 40)
        self.pushButton_open.clicked.connect(self.open_camera)

        self.pushButton_close = QtWidgets.QPushButton('Close Camera', self)
        self.pushButton_close.setGeometry(10, 60, 200, 40)
        self.pushButton_close.clicked.connect(self.close_camera)

        self.pushButton_take_picture = QtWidgets.QPushButton('Take Picture', self)
        self.pushButton_take_picture.setGeometry(10, 110, 200, 40)
        self.pushButton_take_picture.clicked.connect(self.take_picture)
        
        self.video_label = QtWidgets.QLabel(self)
        self.video_label.setGeometry(220, 10, 560, 560)
        self.video_label.setStyleSheet("QLabel { background-color : white; }")
        self.video_label.setScaledContents(True)
        
        self.answer_label = QtWidgets.QLabel(self)
        self.answer_label.setGeometry(800, 10, 560, 560)
        self.answer_label.setStyleSheet("QLabel { background-color : white; }")
        self.answer_label.setScaledContents(True)
        
        self.result_label = QtWidgets.QLabel('Result: ', self)
        self.result_label.setGeometry(10, 160, 200, 40)
        self.result_label.setStyleSheet("QLabel { font-size: 20px; }")
        
        self.detector = WorkDetect()
        self.detector.update_answer_image.connect(self.set_answer_image)
        self.detector.update_result.connect(self.set_result)
        self.detector.start()
        
        self.cam = CamWrapper()
        self.cam.start()
        
        self.video_thread = VideoThread()
        self.video_thread.update_frame.connect(self.set_image)
        self.video_thread.start()
        
    def open_camera(self):
        if not self.cam.is_bind:
            self.cam.bind()
            self.cam.start()
        print('open_cam: ', self.cam.is_bind)
        
    def close_camera(self):
        if self.cam.is_bind:
            self.cam.is_bind = False
            self.cam.stop()
            self.video_thread.stop()
            self.video_label.clear()
        print('off_cam: ', self.cam.is_bind)
        
    def take_picture(self):
        if not self.detector.start_sig:
            self.detector.start_sig = True
        
            current_time = time.strftime("%Y%m%d_%H%M%S")
            file_name = f"picture_{current_time}.jpg"

            pixmap = self.video_label.pixmap()
            if pixmap:
                pixmap.save(file_name, "JPG")
                print(f"Saved image: {file_name}")
            else:
                print("No image to save.")
            
    @pyqtSlot(QImage)
    def set_image(self, image):
        self.video_label.setPixmap(QPixmap.fromImage(image))
    
    @pyqtSlot(QImage)
    def set_answer_image(self, image):
        self.answer_label.setPixmap(QPixmap.fromImage(image))
    
    @pyqtSlot(str)
    def set_result(self, result):
        self.result_label.setText(f"Result: {result}")
    
    def closeEvent(self, event):
        self.video_thread.stop()
        self.cam.stop()
        self.detector.stop()
        event.accept()
        QtWidgets.QApplication.quit()

class CamWrapper(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.is_bind = False
        self.frame = None
        self.camera_list = ['02G93473019']
        self.cam = MvCamera()
        self.CAM_PARSER_FPS = 10
        self.prev_time = 0

    def run(self):
        fps_delay = 1 / self.CAM_PARSER_FPS
        while True:
            if not self.is_bind:
                time.sleep(0.02)
            if self.is_bind:
                if self.frame is not None:
                    if time.time() - self.prev_time >= fps_delay:
                        if len(self.frame.shape) == 2:
                            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BAYER_BG2BGR)
                        window.detector.buffer.append(self.frame)
                        self.prev_time = time.time()
            time.sleep(0.001)

    def bind(self):
        SDKVersion = MvCamera.MV_CC_GetSDKVersion()
        print("SDKVersion[0x%x]" % SDKVersion)
        deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        print('RET >> ', ret)
        if ret != 0:
            print("enum devices fail! ret[0x%x]" % ret)
            sys.exit()
        if deviceList.nDeviceNum == 0:
            print("find no device!")
            sys.exit()
        print("Find %d devices!" % deviceList.nDeviceNum)
        for device in range(0, deviceList.nDeviceNum):
            mvcc_dev_info = cast(deviceList.pDeviceInfo[device], POINTER(MV_CC_DEVICE_INFO)).contents
            if mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
                strModeName = "".join(chr(per) for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName)
                print("device model name: %s" % strModeName)
                strSerialNumber = "".join(chr(per) for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber if per != 0)
                print("user serial number: %s" % strSerialNumber)
            if strSerialNumber in self.camera_list:
                stDeviceList = cast(deviceList.pDeviceInfo[device], POINTER(MV_CC_DEVICE_INFO)).contents
        ret = self.cam.MV_CC_CreateHandle(stDeviceList)
        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        ret = self.cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_Mono8)
        ret = self.cam.MV_CC_SetEnumValue('ExposureMode', 0)
        ret = self.cam.MV_CC_SetFloatValue('ExposureTime',5000.0000)
        stParam = MVCC_INTVALUE()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
        ret = self.cam.MV_CC_GetIntValue("PayloadSize", stParam)
        nPayloadSize = stParam.nCurValue
        ret = self.cam.MV_CC_StartGrabbing()
        self.data_buf = (c_ubyte * nPayloadSize)()
        try:
            hThreadHandle = threading.Thread(target=self.work_thread, args=(self.cam, self.data_buf, nPayloadSize))
            hThreadHandle.start()
        except:
            print('error: unable to start cam work thread')

    def work_thread(self, cam=0, pData=0, nDataSize=0):
        stFrameInfo = MV_FRAME_OUT_INFO_EX()
        memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
        while True:
            ret = cam.MV_CC_GetOneFrameTimeout(pData, nDataSize, stFrameInfo, 1000)
            if ret == 0:
                aimage = np.asarray(pData)
                simage = aimage.reshape((3648, 5472))
                frame_bgr = cv2.cvtColor(simage, cv2.COLOR_BAYER_RG2RGB)
                self.frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                self.frame = simage
                if self.frame is not None:
                    self.is_bind = True
            time.sleep(0.001)

    def stop(self):
        if self.is_bind:
            self.is_bind = False
            self.cam.MV_CC_StopGrabbing()
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()

class YoloDetector(QObject, threading.Thread):
    def __init__(self):
        QObject.__init__(self)
        threading.Thread.__init__(self)
        self.buffer = deque()

        self.net = cv2.dnn.readNet(weights_path, cfg_path)
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i-1] for i in self.net.getUnconnectedOutLayers()]
        with open(names_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

    def warpImg(self, box, frame):
        if len(box) != 4:
            return frame  # 유효하지 않은 경우 원본 프레임 반환

        sum_approx = box.sum(axis=1)
        diff_approx = np.diff(box, axis=1)

        topLeft = box[np.argmin(sum_approx)]
        bottomRight = box[np.argmax(sum_approx)]
        topRight = box[np.argmin(diff_approx)]
        bottomLeft = box[np.argmax(diff_approx)]

        pts1 = np.float32([topLeft, topRight, bottomLeft, bottomRight])

        width_bottom = abs(bottomRight[0] - bottomLeft[0])
        width_top = abs(topRight[0] - topLeft[0])
        height_right = abs(topRight[1] - bottomRight[1])
        height_left = abs(topLeft[1] - bottomLeft[1])

        pcb_width = max(width_bottom, width_top)
        pcb_height = max(height_right, height_left)

        pts2 = np.float32([[0, 0], [pcb_width, 0], [0, pcb_height], [pcb_width, pcb_height]])
    
        mtrx = cv2.getPerspectiveTransform(pts1, pts2)
        warp_frame = cv2.warpPerspective(frame, mtrx, (pcb_width, pcb_height))

        return warp_frame

    def warp(self, frame):
        screenY, screenX = frame.shape[:2]
        resized_frame = cv2.resize(frame, dsize=(screenX // 12, screenY // 12), interpolation=cv2.INTER_AREA)

        enhanced_frame = cv2.convertScaleAbs(resized_frame, alpha=2.0, beta=50)
        gray_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)
        hist_equalized_frame = cv2.equalizeHist(gray_frame)

        _, binary_frame = cv2.threshold(hist_equalized_frame, 127, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((10, 10), np.uint8)
        morph_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(morph_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cont in contours:
            if cv2.contourArea(cont) > 100:
                rect = cv2.minAreaRect(cont)
                box = cv2.boxPoints(rect)
                box = np.intp(box) * 12
                result_frame = self.warpImg(box, frame)

        return result_frame

    def detect(self, frame):
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.05:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
        detected_objects = []
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                detected_objects.append((x, y, w, h, label))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({x},{y},{w},{h})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame, detected_objects

    def detect_qr_code(self, frame):
        detector = cv2.QRCodeDetector()
        data, bbox, _ = detector.detectAndDecode(frame)
        if bbox is not None:
            bbox = bbox.astype(int)
            for i in range(len(bbox[0])):
                cv2.line(frame, tuple(bbox[0][i]), tuple(bbox[0][(i+1) % len(bbox[0])]), (255, 0, 0), 3)
        else:
            print("QR 코드가 인식되지 않았습니다.")
        return data, frame

class WorkDetect(QObject, threading.Thread):
    update_answer_image = pyqtSignal(QImage)
    update_result = pyqtSignal(str)
    
    def __init__(self):
        QObject.__init__(self)
        threading.Thread.__init__(self)
        self.buffer = deque(maxlen=10)
        self.start_sig = False
        self.running = True
        self.detector = YoloDetector()

    def run(self):
        while self.running:
            if self.buffer:
                print('detector_buffer: ', len(self.buffer))
                frame = self.buffer.popleft()
                
                qr_data, frame_with_qr = self.detector.detect_qr_code(frame)
                if qr_data:
                    answer_image_path, answer_coordinates = self.load_answer_data(qr_data)
                    if answer_image_path:
                        answer_image = cv2.imread(answer_image_path)
                        if answer_image is not None:
                            rgb_answer_image = cv2.cvtColor(answer_image, cv2.COLOR_BGR2RGB)
                            for coord in answer_coordinates:
                                cv2.rectangle(rgb_answer_image, coord[0], coord[1], (0, 255, 0), 2)
                            h, w, ch = rgb_answer_image.shape
                            bytes_per_line = ch * w
                            qt_answer_image = QImage(rgb_answer_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                            self.update_answer_image.emit(qt_answer_image)
                        else:
                            print(f"Image file not found or cannot be read: {answer_image_path}")
                    else:
                        print(f"Coordinates file not found for board: {qr_data}")
                
                warped_frame = self.detector.warp(frame_with_qr)
                detected_frame, detected_objects = self.detector.detect(warped_frame)
                result = self.compare_objects_with_answer(detected_objects, answer_coordinates, detected_frame)
                self.update_result.emit(result)
                window.video_thread.buffer.append(detected_frame)
                
                self.start_sig = False
            time.sleep(0.002)

    def stop(self):
        self.running = False

    def load_answer_data(self, qr_data):
        board_name = qr_data.strip()
        answer_image_path = f'answers/{board_name}.bmp'
        coordinates_path = f'answers/{board_name}.txt'
        
        if not os.path.exists(answer_image_path):
            print(f"Image file not found: {answer_image_path}")
            return None, None
        
        if not os.path.exists(coordinates_path):
            print(f"Coordinates file not found: {coordinates_path}")
            return answer_image_path, []
        
        answer_coordinates = []
        with open(coordinates_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        key, values = line.split(' = ')
                        coords = values.split(',')
                        if len(coords) == 4:
                            x1, y1, w, h = map(int, coords)
                            x2, y2 = x1 + w, y1 + h
                            answer_coordinates.append(((x1, y1), (x2, y2)))
                        else:
                            print(f"Invalid format line in coordinates file: {line}")
                    except ValueError:
                        print(f"Invalid line in coordinates file: {line}")
    
        return answer_image_path, answer_coordinates

    def compare_objects_with_answer(self, detected_objects, answer_coordinates, frame):
        all_matched = True
        detected_bbox_list = []
        for detected in detected_objects:
            x, y, w, h, label = detected
            detected_bbox = (x, y, x + w, y + h)
            detected_bbox_list.append(detected_bbox)
            match_found = False
            for coord in answer_coordinates:
                x1, y1 = coord[0]
                x2, y2 = coord[1]
                answer_bbox = (x1, y1, x2, y2)
                if self.iou(detected_bbox, answer_bbox) > 0.5:
                    match_found = True
                    break
            color = (0, 255, 0) if match_found else (0, 0, 255)
            if not match_found:
                all_matched = False
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} ({x},{y},{w},{h})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        for coord in answer_coordinates:
            x1, y1 = coord[0]
            x2, y2 = coord[1]
            answer_bbox = (x1, y1, x2, y2)
            match_found = False
            for detected_bbox in detected_bbox_list:
                if self.iou(detected_bbox, answer_bbox) > 0.5:
                    match_found = True
                    break
            if not match_found:
                all_matched = False
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"Missing", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return "PASS" if all_matched else "FAIL"

    def iou(self, box1, box2):
        x1, y1, x2, y2 = box1
        x1g, y1g, x2g, y2g = box2

        xi1 = max(x1, x1g)
        yi1 = max(y1, y1g)
        xi2 = min(x2, x2g)
        yi2 = min(y2, y2g)
        inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)

        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2g - x1g) * (y2g - y1g)
        union_area = box1_area + box2_area - inter_area

        iou = inter_area / union_area
        return iou

if __name__ == '__main__':
    cfg_path = '/home/ase314b/바탕화면/pca_vision_system/yolo_test/yolo-obj.cfg'
    weights_path = '/home/ase314b/바탕화면/pca_vision_system/yolo_test/backup/pca_system.weights'
    data_path = '/home/ase314b/바탕화면/pca_vision_system/yolo_test/data/obj.data'
    names_path = '/home/ase314b/바탕화면/pca_vision_system/yolo_test/data/obj.names'
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())