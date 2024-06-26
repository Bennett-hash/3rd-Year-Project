import cv2
import ctypes
import os
import shutil
import random
import sys
import threading
import time
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

CONF_THRESH = 0.5
IOU_THRESHOLD = 0.4

def get_img_path_batches(batch_size, img_dir):
    ret = []
    batch = []
    for root, dirs, files in os.walk(img_dir):
        for name in files:
            if len(batch) == batch_size:
                ret.append(batch)
                batch = []
            batch.append(os.path.join(root, name))
    if len(batch) > 0:
        ret.append(batch)
    return ret


def plot_one_box(x, img, id, label=None, line_thickness=None):
    color = get_color(id)
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        full_label = f"{label}"
        t_size = cv2.getTextSize(full_label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, full_label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        
def get_color(id):
    np.random.seed(id)
    color = [np.random.randint(0, 255) for _ in range(3)]
    return color

class YoLov8TRT(object):
    def __init__(self, engine_file_path):
        # Create a Context on this device,
        self.ctx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)
        
        # Initialize DeepSORT Tracker
        cfg_deep = get_config()
        cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
        self.deepsort = DeepSort(
            cfg_deep.DEEPSORT.REID_CKPT,
            max_dist=cfg_deep.DEEPSORT.MAX_DIST,
            max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
            max_age=cfg_deep.DEEPSORT.MAX_AGE,
            n_init=cfg_deep.DEEPSORT.N_INIT,
            nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
            use_cuda=True
        )
        
        self.vehicle_counters = {category: {'entering': 0, 'leaving': 0} for category in categories}
        self.last_positions = {}

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            print("bingding:", binding, engine.get_binding_shape(binding))
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = engine.max_batch_size
        
    def display_vehicle_counts(self, image, vehicle_counters):
        # Starting position for displaying text
        text_pos = (10, 100)
        line_height = 30
    
        for category, counts in vehicle_counters.items():
            entering = counts['entering']
            leaving = counts['leaving']
            info_text = f"{category} - Entering: {entering}, Leaving: {leaving}"
        
            # Display the text on the frame
            cv2.putText(image, info_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (245, 20, 20), 4)
            # Move to the next line for the next category
            text_pos = (text_pos[0], text_pos[1] + line_height)

    def infer(self, image):
        threading.Thread.__init__(self)
        # Make self the active context, pushing it on top of the context stack.
        self.ctx.push()
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        # Do image preprocess
        batch_image_raw = []
        batch_origin_h = []
        batch_origin_w = []
        batch_input_image = np.empty(shape=[1, 3, self.input_h, self.input_w])

        input_image, image_raw, origin_h, origin_w = self.preprocess_image(image)
        batch_image_raw.append(image_raw)
        batch_origin_h.append(origin_h)
        batch_origin_w.append(origin_w)
        np.copyto(batch_input_image, input_image)
        batch_input_image = np.ascontiguousarray(batch_input_image)

        # Copy input image to host buffer
        np.copyto(host_inputs[0], batch_input_image.ravel())
        start = time.time()
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        context.execute_async(
            batch_size=self.batch_size, bindings=bindings, stream_handle=stream.handle
        )
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        # Synchronize the stream
        stream.synchronize()
        end = time.time()

        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        # Here we use the first row of output in that batch_size = 1
        output = host_outputs[0]

        # Do postprocess
        result_boxes, result_scores, result_classid = self.post_process(
            output[0:38001], batch_origin_h[0], batch_origin_w[0]
        )

        num_of_objects = len(result_classid)
        
        # Define an entry/exit line at a specific y-coordinate.
        ENTRY_EXIT_LINE_Y = 270

        # Convert detections to DeepSORT format
        xywhs = self.xyxy_to_xywhs(result_boxes)
        confss = np.array(result_scores)

        # Update DeepSORT tracker with the current frame, detections and object IDs
        outputs = self.deepsort.update(xywhs, confss, result_classid, image_raw)

        # Use DeepSORT outputs for tracking
        for output in outputs:
            bbox = output[:4]  # x1, y1, x2, y2
            track_id = output[4]
            object_idx = int(output[5])  # Assuming this is the object class index
            class_label = categories[object_idx]
            centroid_y = (bbox[1] + bbox[3]) / 2  # Calculate the centroid y-coordinate of the bbox

            # Determine if the vehicle is entering or leaving
            if track_id in self.last_positions:
                if self.last_positions[track_id] < ENTRY_EXIT_LINE_Y <= centroid_y:
                    # Vehicle moving from top to bottom; count as entering
                    self.vehicle_counters[class_label]['entering'] += 1
                elif self.last_positions[track_id] > ENTRY_EXIT_LINE_Y >= centroid_y:
                    # Vehicle moving from bottom to top; count as leaving
                    self.vehicle_counters[class_label]['leaving'] += 1

            # Update the last known position of the vehicle
            self.last_positions[track_id] = centroid_y

            label = f"{class_label}"

            plot_one_box(bbox, image_raw, track_id, label=label)
            
        # Define the color and thickness of the line
        line_color = (0, 255, 0)  # Green line
        line_thickness = 5

        # Draw the ENTRY_EXIT_LINE_Y on the image
        cv2.line(image_raw, (0, ENTRY_EXIT_LINE_Y), (image_raw.shape[1], ENTRY_EXIT_LINE_Y), line_color, line_thickness)
            
        self.display_vehicle_counts(image_raw, self.vehicle_counters)

        return image_raw, end - start, num_of_objects

    def xyxy_to_xywhs(self, bboxes):
        xywhs = []
        for x1, y1, x2, y2 in bboxes:
            x_c = (x1 + x2) / 2
            y_c = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            xywhs.append([x_c, y_c, w, h])
        return np.array(xywhs)

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()

    def get_raw_image(self, image_path_batch):
        for img_path in image_path_batch:
            yield cv2.imread(img_path)

    def get_raw_image_zeros(self, image_path_batch=None):
        for _ in range(self.batch_size):
            yield np.zeros([self.input_h, self.input_w, 3], dtype=np.uint8)

    def preprocess_image(self, raw_bgr_image):
        image_raw = raw_bgr_image
        h, w, c = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        # Calculate widht and height and paddings
        r_w = self.input_w / w
        r_h = self.input_h / h
        if r_h > r_w:
            tw = self.input_w
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((self.input_h - th) / 2)
            ty2 = self.input_h - th - ty1
        else:
            tw = int(r_h * w)
            th = self.input_h
            tx1 = int((self.input_w - tw) / 2)
            tx2 = self.input_w - tw - tx1
            ty1 = ty2 = 0
        # Resize the image with long side while maintaining ratio
        image = cv2.resize(image, (tw, th))
        # Pad the short side with (128,128,128)
        image = cv2.copyMakeBorder(
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, None, (128, 128, 128)
        )
        image = image.astype(np.float32)
        # Normalize to [0,1]
        image /= 255.0
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        image = np.ascontiguousarray(image)
        return image, image_raw, h, w

    def xywh2xyxy(self, origin_h, origin_w, x):
        y = np.zeros_like(x)
        r_w = self.input_w / origin_w
        r_h = self.input_h / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0]
            y[:, 2] = x[:, 2]
            y[:, 1] = x[:, 1] - (self.input_h - r_w * origin_h) / 2
            y[:, 3] = x[:, 3] - (self.input_h - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - (self.input_w - r_h * origin_w) / 2
            y[:, 2] = x[:, 2] - (self.input_w - r_h * origin_w) / 2
            y[:, 1] = x[:, 1]
            y[:, 3] = x[:, 3]
            y /= r_h

        return y

    def post_process(self, output, origin_h, origin_w):
        # Get the num of boxes detected
        num = int(output[0])
        # Reshape to a two dimentional ndarray
        pred = np.reshape(output[1:], (-1, 38))[:num, :]
        # Do nms
        boxes = self.non_max_suppression(
            pred, origin_h, origin_w, conf_thres=CONF_THRESH, nms_thres=IOU_THRESHOLD
        )
        result_boxes = boxes[:, :4] if len(boxes) else np.array([])
        result_scores = boxes[:, 4] if len(boxes) else np.array([])
        result_classid = boxes[:, 5] if len(boxes) else np.array([])
        return result_boxes, result_scores, result_classid

    def bbox_iou(self, box1, box2, x1y1x2y2=True):
        if not x1y1x2y2:
            # Transform from center and width to exact coordinates
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            # Get the coordinates of bounding boxes
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        # Get the coordinates of the intersection rectangle
        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)
        # Intersection area
        inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * np.clip(
            inter_rect_y2 - inter_rect_y1 + 1, 0, None
        )
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

        return iou

    def non_max_suppression(
        self, prediction, origin_h, origin_w, conf_thres=0.5, nms_thres=0.4
    ):
        # Get the boxes that score > CONF_THRESH
        boxes = prediction[prediction[:, 4] >= conf_thres]
        # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
        boxes[:, :4] = self.xywh2xyxy(origin_h, origin_w, boxes[:, :4])
        # clip the coordinates
        boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_w - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_h - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_h - 1)
        # Object confidence
        confs = boxes[:, 4]
        # Sort by the confs
        boxes = boxes[np.argsort(-confs)]
        # Perform non-maximum suppression
        keep_boxes = []
        while boxes.shape[0]:
            large_overlap = (
                self.bbox_iou(np.expand_dims(boxes[0, :4], 0), boxes[:, :4]) > nms_thres
            )
            label_match = boxes[0, -1] == boxes[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            keep_boxes += [boxes[0]]
            boxes = boxes[~invalid]
        boxes = np.stack(keep_boxes, 0) if len(keep_boxes) else np.array([])
        return boxes


class inferThread(threading.Thread):
    def __init__(self, yolov8_wrapper):
        threading.Thread.__init__(self)
        self.yolov8_wrapper = yolov8_wrapper
        self.out = None

        self.cap = cv2.VideoCapture(
            "/home/jetson/yolo/NanoYOLO/testV.mp4"
        )

        # Check if the video file was successfully loaded
        if not self.cap.isOpened():
            print("Error opening video file")

    def run(self):
        prev_frame_time = 0
        new_frame_time = 0
        total_number_of_objects = 0
        
        # Initialization of video writer
        self.out = cv2.VideoWriter('/home/jetson/yolo/tensorrtx/yolov8/runs/detect/output_video.avi', 
                                   cv2.VideoWriter_fourcc(*'XVID'), 
                                   30, (1920, 1080))

        while True:
            ret, frame = self.cap.read()

            if not ret:
                break

            img = cv2.resize(frame, (1920, 1080))

            new_frame_time = time.time()
            result, use_time, number_of_objects = self.yolov8_wrapper.infer(img)

            total_number_of_objects += number_of_objects
            # Calculate FPS of complete processing time of one frame (not just inference)
            fps = int(1 / (new_frame_time - prev_frame_time))
            prev_frame_time = new_frame_time
            # TODO
            print(
                "inference: time->{:.2f}ms, fps: {:.2f}, total fps: {:.2f}, frame obj: {}, total obj: {}".format(
                    use_time * 1000, 1 / (use_time), fps, number_of_objects, total_number_of_objects
                )
            )

            # Inference FPS:
            fps_infer = 1 / (use_time)
            fps_infer = format(fps_infer, '.2f')
            fps = str(fps_infer) + " FPS"
            yolo_model_name = "YOLOv8n TRT"
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(
                result,
                yolo_model_name,
                (10, 50),
                font,
                1,
                (245, 20, 20),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                result,
                fps,
                (640 - 200, 50),
                font,
                1,
                (20, 20, 245),
                2,
                cv2.LINE_AA,
            )
            #cv2.imshow("Recognition result", result)
            
            # Writing the processed frame to the video
            self.out.write(result)

            
            #if cv2.waitKey(1) & 0xFF == ord("q"):
                #break


if __name__ == "__main__":
    PLUGIN_LIBRARY = "/home/jetson/yolo/tensorrtx/yolov8/build/libmyplugins.so"
    engine_file_path = "/home/jetson/yolo/tensorrtx/yolov8/build/v8n.engine"

    if len(sys.argv) > 1:
        engine_file_path = sys.argv[1]
    if len(sys.argv) > 2:
        PLUGIN_LIBRARY = sys.argv[2]

    ctypes.CDLL(PLUGIN_LIBRARY)

    # load coco labels

    categories = ["Truck", "Sedan", "SUV", "Motorbike", "Bus"]

    if os.path.exists("output/"):
        shutil.rmtree("output/")
    os.makedirs("output/")
    # a YoLov8TRT instance
    yolov8_wrapper = YoLov8TRT(engine_file_path)
    try:
        print("batch size is", yolov8_wrapper.batch_size)
        # create a new thread to do inference
        thread1 = inferThread(yolov8_wrapper)
        thread1.start()
        thread1.join()
    finally:
        # destroy the instance
        # Releasing the resources
        if thread1.out:
            thread1.out.release()
        cv2.destroyAllWindows()
        yolov8_wrapper.destroy()
