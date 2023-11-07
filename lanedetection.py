import os
from flask import Flask, render_template, request, send_from_directory
import numpy as np
import cv2
from moviepy.editor import VideoFileClip

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["OUTPUT_FOLDER"] = "static"
app.config["ALLOWED_EXTENSIONS"] = {"mp4"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

def process_video(input_path, output_path):
    input_video = VideoFileClip(input_path, audio=False)
    processed = input_video.fl_image(frame_processor)
    output_path = os.path.join(app.config["OUTPUT_FOLDER"], output_path)
    processed.write_videofile(output_path, audio=False)
    return output_path

def region_selection(image):
    #create an array of the same size as of the input image 
    mask = np.zeros_like(image)   
    # if you pass an image with more then one channel
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    # our image only has one channel so it will go under "else"
    else:
          # color of the mask polygon (white)
        ignore_mask_color = 255
    # creating a polygon to focus only on the road in the picture
    # we have created this polygon in accordance to how the camera was placed
    rows, cols = image.shape[:2]
    bottom_left  = [cols * 0.1, rows * 0.95]
    top_left     = [cols * 0.4, rows * 0.6]
    bottom_right = [cols * 0.9, rows * 0.95]
    top_right    = [cols * 0.6, rows * 0.6]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    # filling the polygon with white color and generating the final mask
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    # performing Bitwise AND on the input image and mask to get only the edges on the road
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def hough_transform(image):
    # Distance resolution of the accumulator in pixels.
    rho = 1             
    # Angle resolution of the accumulator in radians.
    theta = np.pi/180   
    # Only lines that are greater than threshold will be returned.
    threshold = 20      
    # Line segments shorter than that are rejected.
    minLineLength = 20  
    # Maximum allowed gap between points on the same line to link them
    maxLineGap = 500    
    # function returns an array containing dimensions of straight lines 
    # appearing in the input image
    return cv2.HoughLinesP(image, rho = rho, theta = theta, threshold = threshold,
                           minLineLength = minLineLength, maxLineGap = maxLineGap)

def average_slope_intercept(lines):
    left_lines    = [] #(slope, intercept)
    left_weights  = [] #(length,)
    right_lines   = [] #(slope, intercept)
    right_weights = [] #(length,)
     
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            # calculating slope of a line
            slope = (y2 - y1) / (x2 - x1)
            # calculating intercept of a line
            intercept = y1 - (slope * x1)
            # calculating length of a line
            length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
            # slope of left lane is negative and for right lane slope is positive
            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))
    # 
    left_lane  = np.dot(left_weights,  left_lines) / np.sum(left_weights)  if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
    return left_lane, right_lane

def pixel_points(y1, y2, line):
    if line is None:
        return None
    slope, intercept = line
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)
    return ((x1, y1), (x2, y2))

def lane_lines(image, lines):
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0]
    y2 = y1 * 0.6
    left_line  = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)
    return left_line, right_line

def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=12):
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line,  color, thickness)
    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)

def frame_processor(image):
    # convert the RGB image to Gray scale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # applying gaussian Blur which removes noise from the image 
    # and focuses on our region of interest
    # size of gaussian kernel
    kernel_size = 5
    # Applying gaussian blur to remove noise from the frames
    blur = cv2.GaussianBlur(grayscale, (kernel_size, kernel_size), 0)
    # first threshold for the hysteresis procedure
    low_t = 50
    # second threshold for the hysteresis procedure 
    high_t = 150
    # applying canny edge detection and save edges in a variable
    edges = cv2.Canny(blur, low_t, high_t)
    # since we are getting too many edges from our image, we apply 
    # a mask polygon to only focus on the road
    # Will explain Region selection in detail in further steps
    region = region_selection(edges)
    # Applying hough transform to get straight lines from our image 
    # and find the lane lines
    # Will explain Hough Transform in detail in further steps
    hough = hough_transform(region)
    #lastly we draw the lines on our resulting frame and return it as output 
    result = draw_lane_lines(image, lane_lines(image, hough))
    return result

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "video" not in request.files:
            return "No video part"

        video = request.files["video"]

        if video.filename == "":
            return "No selected video file"

        if video and allowed_file(video.filename):
            video_path = os.path.join(app.config["UPLOAD_FOLDER"], "input.mp4")
            video.save(video_path)

            processed_video_path = process_video(video_path, "processed_video.mp4")
            return render_template("upload.html", processed_video=processed_video_path)

    return render_template("upload.html")

@app.route("/process", methods=["POST"])
def process():
    if request.method == "POST":
        if "video" not in request.files:
            return "No video part"

        video = request.files["video"]

        if video.filename == "":
            return "No selected video file"

        if video and allowed_file(video.filename):
            video_path = os.path.join(app.config["UPLOAD_FOLDER"], "input.mp4")
            video.save(video_path)

            processed_video_path = process_video(video_path, "processed_video.mp4")
            return render_template("upload.html", processed_video=processed_video_path)

    return render_template("upload.html")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    if not os.path.exists(app.config["UPLOAD_FOLDER"]):
        os.makedirs(app.config["UPLOAD_FOLDER"])
    app.run(debug=True)
