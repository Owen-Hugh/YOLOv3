import cv2
import time
import numpy as np

Wide = 320
confThreshold = 0.5 # 置信度阈值
nmsThreshold = 0.3 # 非极大值抑制的阈值
ftime= 0

cap = cv2.VideoCapture(0)

print("正在加载标签...")
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

print("正在加载网络权重...")

# 在这里加个class选择yolo

modelConfiguration = 'yolov3-tiny.cfg'
modelWeights = 'yolov3-tiny.weights'

# 创建网络
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)  # gpu
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # cpu
print("加载完毕！")

def findObjects(outputs, img):
    """
    查找对象
    :param outputs:输出
    :param img:图像
    :return:x,y,w,h
    """
    hT, wT, cT = img.shape
    bbox = [] # x,y,h,w
    classIds = [] # 类, id
    confs = [] # 置信度

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores) # 置信度最大的索引
            confidence = scores[classId] # 置信度最大的值
            if confidence > confThreshold:
                w,h = int(det[2]*wT) , int(det[3]*hT)
                x,y = int(det[0]*wT - w/2),int(det[1]*hT - h/2) # 中心点坐标转左上角
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)

    for i in indices:
        i = i[0]
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(img,(x,y),(x+w,y+h),color=(0,0xff,0xc5),thickness=2)
        cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                    (x,abs(y-10)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0xa1),2)

while True:
    sss, img = cap.read()
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (Wide, Wide), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    outputNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # print(outputNames)
    outputs = net.forward(outputNames)  # 边界框、对象名称、ID...
    findObjects(outputs,img)

    ltime = time.time()
    fps = 1 / (ltime - ftime)
    ftime = time.time()  # 时间读取
    cv2.putText(img, 'FPS: ' + str(int(fps)), (25, 50), cv2.FONT_HERSHEY_PLAIN, 2, (100, 100, 255), 2)

    cv2.imshow('img', img)
    if ord('q') == cv2.waitKey(1):
        break