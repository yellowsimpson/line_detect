import cv2
import numpy as np

def apply_color_filter(image):
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define color range for yellow (typical highway lane color)
    lower_yellow = np.array([18, 94, 140])
    upper_yellow = np.array([48, 255, 255])
    
    # Define color range for white (typical lane color)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([255, 255, 255])
    
    # Create masks for yellow and white lanes
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    
    # Combine the masks
    combined_mask = cv2.bitwise_or(mask_yellow, mask_white)
    
    # Apply the mask to the original image
    filtered_image = cv2.bitwise_and(image, image, mask=combined_mask)
    
    return filtered_image

def canny_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:  # 왼쪽 차선
            left_fit.append((slope, intercept))
        else:  # 오른쪽 차선
            right_fit.append((slope, intercept))
    
    left_fit_average = np.average(left_fit, axis=0) if left_fit else None
    right_fit_average = np.average(right_fit, axis=0) if right_fit else None
    
    left_line = create_coordinates(image, left_fit_average) if left_fit_average is not None else None
    right_line = create_coordinates(image, right_fit_average) if right_fit_average is not None else None
    
    return np.array([left_line, right_line])

def create_coordinates(image, line_parameters):
    if line_parameters is None:
        return None
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def process_frame(frame):
    # Apply color filter to focus on lane lines
    filtered_frame = apply_color_filter(frame)
    
    # Detect edges
    canny_image = canny_edge_detection(filtered_frame)
    
    # Focus on the region of interest
    cropped_image = region_of_interest(canny_image)
    
    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    
    # Average lines for smoother lane detection
    averaged_lines = average_slope_intercept(frame, lines)
    
    # Display the lines on the original frame
    line_image = display_lines(frame, averaged_lines)
    
    # Combine the original frame with the detected lines
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    
    return combo_image

def main():
    video_path = "thesis/1.AVI"  # Replace with your video file path
    cap = cv2.VideoCapture(video_path)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(frame)
        
        # Show the processed frame
        cv2.imshow("Lane Detection", processed_frame)
        
        # Save the processed frame to a video
        out.write(processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
