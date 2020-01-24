#-----------------------------------------------------------
# import the necessary packages
#-----------------------------------------------------------
import numpy as np
import argparse
import imutils
import time
import cv2
import os
 
#-----------------------------------------------------------
# construct the argument parse and parse the arguments
#-----------------------------------------------------------
ap = argparse.ArgumentParser()

ap.add_argument("-y", "--yolo", required=True, help="base path to YOLO directory")
ap.add_argument("-d", "--device", type=int, default=0, help="device number for the detection(usually 0 for front cam and 1 for rear cam")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())


#-----------------------------------------------------------
# load the COCO class labels our YOLO model was trained on
#-----------------------------------------------------------
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
 
    
#-----------------------------------------------------------
# initialize a list of colors to represent each possible class label
#-----------------------------------------------------------
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
 
    
#-----------------------------------------------------------
# derive the paths to the YOLO weights and model configuration
#-----------------------------------------------------------
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])
 
    
#-----------------------------------------------------------
# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
#-----------------------------------------------------------
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


#-----------------------------------------------------------
# initialize the video stream, pointer to output video file, and
# frame dimensions
#-----------------------------------------------------------
cap = cv2.VideoCapture(args["device"])
writer = None
(W, H) = (None, None)


print("[INFO] press 'q' button to close the window...")
print("[INFO] device number: ", args["device"])
#-----------------------------------------------------------
# loop over frames from the video file stream
#-----------------------------------------------------------
while True:
    # read the next frame from the file
    ret, frame = cap.read()
    
    
    # if the frame was not grabbed, then we have reached the end
    # of the stream]    
    if not ret:
        print("the frame was not grabbed")
        break
        
    # when you use a front cam, flip it horizontally
    if args["device"] == 0:
        frame = cv2.flip(frame, 1)

    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]


    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    #print("[1]", end-start)
 
    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
    
    start2 = time.time()
    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > args["confidence"]:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)


    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

    # ensure at least one detection exists
    if len(idxs) > 0:
        var_b = 0
        # loop over the indexes we are keeping
        for i in idxs.flatten():
#             print("...")
            var_b = 0
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#             cv2.imshow('Video', frame)
    
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 var_b = 1
#                 break
#         if var_b == 1:
#             break

    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
            
    end2 = time.time()
    #print("[2]", end2-start2, "\n")


# release the file pointers
print("[INFO] cleaning up...")
cap.release()
cv2.destroyAllWindows()