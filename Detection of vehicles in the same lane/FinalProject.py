import pyopencl as cl
import cv2 as cv
import tkinter as tk
from tkinter import filedialog
import numpy as np
import imutils
import math
import config as config


############################### YOLO ##################################
classes = open(config.labels).read().strip().split('\n')

# load YOLO from cv2.dnn
net = cv.dnn.readNetFromDarknet(config.model_config, config.model)
output_layer_names = net.getUnconnectedOutLayersNames()

# Add a global variable to store accumulated detections
accumulated_detections = []
############################ GET VIDEO ################################

root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()

############################ GET DISTANCE ################################

saved_distance = None
last_saved = 0

############################ OPENCL ################################

# Create OpenCL context and command queue
platforms = cl.get_platforms()
devices = platforms[0].get_devices()
context = cl.Context([devices[0]])
queue = cl.CommandQueue(context)

# Load and compile the OpenCL kernel
with open("kernel_function.cl", "r") as file:
    kernel_source = file.read()
program = cl.Program(context, kernel_source).build()


############################ FUNCTIONS ################################

def highlight_lines(vidFrame):
    # Convert BGR to uchar4
    uchar4_vidFrame = cv.cvtColor(vidFrame, cv.COLOR_BGR2BGRA)

    # Set kernel arguments
    width = np.int32(uchar4_vidFrame.shape[1])
    height = np.int32(uchar4_vidFrame.shape[0])

    roi_vertices = np.int32(config.get_roi_vertices(width,height))
    roi_vertices_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=roi_vertices)

    img_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=uchar4_vidFrame)

    result_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, uchar4_vidFrame.nbytes)

    accumulator_left = np.zeros((round(math.sqrt(width * width + height * height)), 180), dtype=np.int32)
    accumulator_left_buf = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                                     hostbuf=accumulator_left)

    accumulator_right = np.zeros((round(math.sqrt(width * width + height * height)), 180), dtype=np.int32)
    accumulator_right_buf = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                                      hostbuf=accumulator_right)

    # Arguments kernel
    kernelName = program.KernelFunction
    kernelName.set_arg(0, img_buf)
    kernelName.set_arg(1, result_buf)
    kernelName.set_arg(2, accumulator_left_buf)
    kernelName.set_arg(3, accumulator_right_buf)
    kernelName.set_arg(4, width)
    kernelName.set_arg(5, height)
    kernelName.set_arg(6, np.int32(config.T1))
    kernelName.set_arg(7, np.int32(config.T2))
    kernelName.set_arg(8, roi_vertices_buf)
    kernelName.set_arg(9, np.int32(config.num_vertices))
    kernelName.set_arg(10, np.int32(config.L_Min_Ang))
    kernelName.set_arg(11, np.int32(config.L_Max_Ang))
    kernelName.set_arg(12, np.int32(config.R_Min_Ang))
    kernelName.set_arg(13, np.int32(config.R_Max_Ang))

    # Launch the kernel
    globalWorkSize = (width, height)
    localWorkSize = None
    kernelEvent = cl.enqueue_nd_range_kernel(queue=queue, kernel=kernelName, global_work_size=globalWorkSize,
                                             local_work_size=localWorkSize)
    # Wait for the kernel to finish
    kernelEvent.wait()

    # Copy the result back to the host
    cl.enqueue_copy(queue, accumulator_left, accumulator_left_buf)
    cl.enqueue_copy(queue, accumulator_right, accumulator_right_buf)

    outputImg = np.empty_like(uchar4_vidFrame)
    cl.enqueue_copy(queue, outputImg, result_buf)

    pt1, pt2 = find_lines_left(accumulator_left)
    pt3, pt4 = find_lines_right(accumulator_right)

    cv.line(vidFrame, pt1, pt2, (255, 0, 0), 1, cv.LINE_AA)
    cv.line(vidFrame, pt3, pt4, (255, 0, 0), 1, cv.LINE_AA)

    img_buf.release()
    result_buf.release()

    return vidFrame, pt1, pt2, pt3, pt4


def find_cars(img, pt1, pt2, pt3, pt4, conf_threshold=0.75, classes_of_interest=None, classes=[], net=net):
    global accumulated_detections

    boxes = []
    confidences = []
    classIDs = []

    H, W = img.shape[:2]

    blob = cv.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = np.vstack(net.forward(output_layer_names))

    for output in outputs:
        scores = output[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        if confidence > conf_threshold and (classes_of_interest is None or classes[classID] in classes_of_interest):
            x, y, w, h = output[:4] * np.array([W, H, W, H])
            p0 = int(x - w // 2), int(y - h // 2)
            p1 = int(x + w // 2), int(y + h // 2)
            boxes.append([*p0, int(w), int(h)])
            confidences.append(round(float(confidence), 2))
            classIDs.append(classID)

    # Apply non-maximum suppression
    indices = cv.dnn.NMSBoxes(boxes, confidences, conf_threshold, conf_threshold - 0.1)

    if len(indices) != 0:
        for i in indices.flatten():
            (x, y, w, h) = (boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3])

            is_inside_lane = is_car_inside_lane(pt1, pt2, pt3, pt4, (x + w // 2), (y + h // 2))

            if is_inside_lane:

                cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                label = f"{classes[classIDs[i]]}: {confidences[i]:.2f}"
                cv.putText(img, label, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                roi_points = np.array([(x, y + (h // 2)), (x + w, y + (h // 2)), (x + w, y + h), (x, y + h)], dtype=np.int32)
                get_distance(img, config.KNOWN_DISTANCE, config.KNOWN_WIDTH, roi_points=roi_points)

            else:
                cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"{classes[classIDs[i]]}: {confidences[i]:.2f}"
                cv.putText(img, label, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return img


def get_distance(img, known_dist, known_width, roi_points):
    global saved_distance
    global last_saved

    mask = np.zeros_like(img)
    cv.fillPoly(mask, [roi_points], (255, 255, 255))
    imgROI = cv.bitwise_and(img, mask)

    gray = cv.cvtColor(imgROI, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)

    edged = cv.Canny(gray, 90, 100)
    cnts = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if cnts:
        for c in cnts:
            marker = cv.minAreaRect(c)

            box = cv.boxPoints(marker)
            box = np.intp(box)

            widthA = np.linalg.norm(box[0] - box[1])
            widthB = np.linalg.norm(box[1] - box[2])
            is_rectangle = abs(box[0][0] - box[3][0]) < 5 and abs(box[1][0] - box[2][0]) < 3

            if 20 < widthA < 90 and 10 < widthB < 40 and 3 < widthA / widthB < 3.5 and is_rectangle:

                measured_width = max(widthA, widthB)
                saved_distance = round((known_dist * known_width) / measured_width, 1)
                last_saved = 0

    if last_saved < 5:
        cv.putText(img, f"{saved_distance} m", (1050, 830), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        last_saved += 1


def process_image(vidFrame):
    vidFrame = cv.resize(vidFrame, (1200, 900))

    vidFrame, pt1, pt2, pt3, pt4 = highlight_lines(vidFrame)

    find_cars(vidFrame, pt1, pt2, pt3, pt4, conf_threshold=config.CONFIDENCE_THRESHOLD,classes_of_interest=config.CLASSES_OF_INTEREST, classes=classes, net=net)

    return vidFrame


################################ HELPER FUNCTIONS ###############################
def find_lines_left(accumulator):
    most_votes = np.max(accumulator)
    most_votes_index = np.argmax(accumulator == most_votes)

    rho = (most_votes_index // 180)
    theta = most_votes_index % 180

    a = math.cos(math.radians(theta))
    b = math.sin(math.radians(theta))
    x0 = a * rho
    y0 = b * rho

    pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
    pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))

    return pt1, pt2


def find_lines_right(accumulator):
    most_votes = np.max(accumulator)
    most_votes_index = np.argmax(accumulator == most_votes)

    rho = (most_votes_index // 180 - 948 / 2)
    theta = most_votes_index % 180

    a = math.cos(math.radians(theta))
    b = math.sin(math.radians(theta))
    x0 = a * rho
    y0 = b * rho

    pt1 = (int(x0 + 1500 * (-b)), int(y0 + 1500 * (a)))
    pt2 = (int(x0 - 1500 * (-b)), int(y0 - 1500 * (a)))

    return pt1, pt2


def is_car_inside_lane(pt1, pt2, pt3, pt4, x, y):
    roi_points = np.array([pt1, pt2, pt3, pt4], dtype=np.int32)
    distance = cv.pointPolygonTest(roi_points, (x, y), False)

    return distance >= 0

#################################### MAIN #####################################
def show_video(filename):
    vidCap = cv.VideoCapture(filename)

    if not vidCap.isOpened():
        print("Video File Not Found")
        exit(-1)

    while True:
        ret, vidFrame = vidCap.read()
        if not ret:
            break

        vidFrame = process_image(vidFrame)
        cv.imshow("Processed Video", vidFrame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    vidCap.release()
    cv.destroyAllWindows()


show_video(file_path)