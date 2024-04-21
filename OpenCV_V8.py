import cv2
import time

# Load video from file
cap = cv2.VideoCapture("/home/jetson/yolo/NanoYOLO/testV.mp4")
# Initialize MOG2 background subtractor with detectShadows set to False
mog = cv2.createBackgroundSubtractorKNN(detectShadows=False)

# Create a 5x5 structuring element with a rectangular shape
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 3))
# Set minimum width and height for vehicle bounding boxes
min_w = 30
min_h = 30
# Set the height for the counting line
line_high = 175
# Set the offset for counting
offset = 20

# Dictionary to hold the last known y-coordinate of each car's centroid
previous_centroids = {}
car_IDs = 0  # Unique identifier for each car
car_tracker = {}  # Tracks each car's position and count status

# Counters for entering and leaving cars
entering_cars = 0
leaving_cars = 0

# Variables for FPS calculation
total_frames = 0
total_time = 0.0

# Resize dimensions
resize_width = 980
resize_height = 540

# Setup video writer to save output
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get video width
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get video height
fps = cap.get(cv2.CAP_PROP_FPS)  # Get video FPS
fourcc = cv2.VideoWriter_fourcc(*'XVID')

out = cv2.VideoWriter("/home/jetson/yolo/OpenCV/OpenCV_Out_V6_close_Kernal_Ellip_3*6.avi", fourcc, fps, (resize_width, resize_height), False)

# Variables for timing each process
times = {"resize": 0.0, "grayscale": 0.0, "gaussian_blur": 0.0, "background_subtraction": 0.0, "erode": 0.0, "dilate": 0.0, "close": 0.0, "find_contours": 0.0, "tracking_counting": 0.0}

ret, frame = cap.read()

# Resize and convert to grayscale as the initial background model
frame = cv2.resize(frame, (resize_width, resize_height))
background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype("float")

# Background update factor
alpha = 0.25

# Create a 5x5 structuring element with a rectangular shape
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 3))

# Threshold for black and white conversion
threshold_value = 35
max_value = 255  # The value to set if the pixel value exceeds the threshold

# Loop through each frame of the video
while True:
    start_time_t = time.time()  # Start time for FPS calculation
    ret, frame = cap.read()# Read a frame from the video

    if ret is True:
        total_frames += 1  # Increment the total frame count

        # Resize frame for processing
        start_time = time.time()
        frame = cv2.resize(frame, (resize_width, resize_height))
        times["resize"] += time.time() - start_time

        # Convert frame to grayscale
        start_time = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        times["grayscale"] += time.time() - start_time

        # Apply Gaussian blur to reduce noise
        start_time = time.time()
        blur = cv2.GaussianBlur(gray, (3, 3), 5)
        times["gaussian_blur"] += time.time() - start_time

        # Apply background subtraction
        start_time = time.time()
        cv2.accumulateWeighted(gray, background, alpha)
        foreground_mask = cv2.absdiff(gray, cv2.convertScaleAbs(background))
        ret, bw_frame = cv2.threshold(foreground_mask, threshold_value, max_value, cv2.THRESH_BINARY)
        times["background_subtraction"] += time.time() - start_time

        # Erode to reduce noise and separate objects
        start_time = time.time()
        erode = cv2.erode(bw_frame, kernel)
        times["erode"] += time.time() - start_time

        # Dilate to restore eroded objects
        start_time = time.time()
        dilate = cv2.dilate(erode, kernel, iterations=5)
        times["dilate"] += time.time() - start_time

        # Close operation to fill in holes
        start_time = time.time()
        close = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
        times["close"] += time.time() - start_time

        # Find contours in the mask
        start_time = time.time()
        contours, h = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        times["find_contours"] += time.time() - start_time

        start_time = time.time()
        # Draw the counting line
        cv2.line(frame, (0, line_high), (width, line_high), (255, 255, 0), 2)

        # Dictionary for storing the current frame's car information
        current_frame_cars = {}

        # Process each detected contour
        for contour in contours:
            # Get bounding box for the contour
            (x, y, w, h) = cv2.boundingRect(contour)
            # Filter out small rectangles
            if not ((w >= min_w) and (h >= min_h)):
                continue
            if (w > min_w) and (h > min_h):
                # Draw bounding box around detected object
                cv2.rectangle(frame, (x, y), (x+w, y + h), (0, 0, 255), 2)
                # Calculate centroid of the bounding box
                cx = int(x + 0.5 * w)
                cy = int(y + 0.5 * h)
                # Draw the centroid
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            # Calculate the centroid using moments
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Determine if the centroid represents a new or existing car
            is_new_car = True
            for car_id, (prev_cx, prev_cy, prev_frame_cy, counted) in car_tracker.items():
                if abs(cx - prev_cx) < 80 and abs(cy - prev_cy) < 40:  # Threshold for matching cars
                    is_new_car = False
                    # Update existing car info
                    # Carry over the previous y-coordinate and counted status
                    current_frame_cars[car_id] = (cx, cy, prev_frame_cy, counted)
                    break

            if is_new_car:
                # Assign new ID and add to current frame cars
                current_frame_cars[car_IDs] = (cx, cy, cy, False)
                car_IDs += 1

         # Update counters based on car movement
        for car_id, (cx, cy, prev_frame_cy, counted) in current_frame_cars.items():
            if (line_high - offset) < cy < (line_high + offset):
                if not counted:  # Check if the car hasn't been counted yet
                    if cy > prev_frame_cy:  # Car moving down
                        entering_cars += 1
                    elif cy < prev_frame_cy:  # Car moving up
                        leaving_cars += 1
                    current_frame_cars[car_id] = (cx, cy, cy, True)  # Mark car as counted

        # Update the car tracker for the next frame
        car_tracker = current_frame_cars.copy()

        times["tracking_counting"] += time.time() - start_time

        # cv2.imshow('video1', frame)
        # cv2.imshow('video2', dst)
        # cv2.imshow('video3', gray)
        # cv2.imshow('video4', blur)
        # cv2.imshow('video5', mask)
        # cv2.imshow('video6', erode)
        # cv2.imshow('video7', dilate)
        # cv2.imshow('video8', close)
        # cv2.putText(frame, 'Vehicle Count:' + str(carnum), (420, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (245, 20, 20), 3)

        # Resize processed frame back to original dimensions before saving
        #frame = cv2.resize(frame, (1920, 1080))

        # Calculate and display FPS for current frame
        end_time = time.time()
        frame_time = end_time - start_time_t
        total_time += frame_time  # Accumulate the processing time

        if frame_time > 0:
            current_fps = 1.0 / frame_time
            cv2.putText(frame, 'FPS: {:.2f}'.format(current_fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        # Display entering and leaving car counts
        cv2.putText(frame, 'Entering:' + str(entering_cars), (420, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.putText(frame, 'Leaving :' + str(leaving_cars), (420, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Save each frame into video
        #out.write(close)

    else:
    # If no frame is captured, break out of the loop
        break

# Calculate average processing time per frame for each process in milliseconds
average_times_ms = {process: (total_time / total_frames) * 1000 for process, total_time in times.items()}

print("Average time taken per frame for each process:")
for process, avg_time_ms in average_times_ms.items():
    print(f"{process}: {avg_time_ms:.4f} ms")

# Calculate the average FPS
if total_time > 0:
    average_fps = total_frames / total_time
    print(f'Average FPS: {average_fps:.2f}')
    print(f'Entering: {entering_cars}')
    print(f'Leaving: {leaving_cars}')
else:
    print('No frames were processed.')

# Release resources
cap.release()
out.release()
#cv2.destroyALLWindows()
