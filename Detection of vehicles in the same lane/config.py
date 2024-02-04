############################### YOLO ##################################
CONFIDENCE_THRESHOLD = 0.60
NMS_THRESHOLD = 0.3
CLASSES_OF_INTEREST = ["car", "truck", "bus", "motorcycle", "bicycle", "van"]

model = 'yolo_files/yolov3.weights'
model_config = 'yolo_files/yolov3.cfg'
labels = 'yolo_files/coco.names'

############################ GET DISTANCE ################################
KNOWN_DISTANCE = 5.0  # m
KNOWN_WIDTH = 85.0  # px

############################ HOUGH ################################
MAX_RHO = 10000

T1 = 30
T2 = 160

L_Min_Ang = 30
L_Max_Ang = 50
R_Min_Ang = 135
R_Max_Ang = 155

num_vertices = 12

def get_roi_vertices(width,height):
    roi_vertices = [
        (0, height * 5 / 6),
        (0, height * 4 / 6),
        (width * 2 / 8, height / 4),
        (width * 6 / 8, height / 4),
        (width, height * 4 / 6),
        (width, height * 5 / 6),
        (width * 3 / 4, height * 5 / 6),
        (width * 5 / 8, height / 2),
        (width * 3 / 8, height / 2),
        (width * 1 / 4, height * 5 / 6),
        (width * 1 / 5, height * 5 / 6),
        (0, height * 5 / 6)
    ]

    return roi_vertices