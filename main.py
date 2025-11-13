import cv2
import numpy as np
import time

# --- Setup for Multiple Colors ---
# Define a list of colors, their HSV ranges, and the BGR color used for the bounding box.
# NOTE: Red often requires two separate HSV ranges because the Hue circle wraps around 0/180.
color_ranges = [
    {
        'name': 'Red',
        'ranges': [
            (np.array([0, 100, 100]), np.array([10, 255, 255])),     # Lower Red Hue
            (np.array([160, 100, 100]), np.array([179, 255, 255]))  # Upper Red Hue
        ],
        'bgr_color': (0, 0, 255) # BGR (Blue=0, Green=0, Red=255)
    },
    {
        'name': 'Green',
        'ranges': [
            (np.array([40, 70, 70]), np.array([80, 255, 255]))
        ],
        'bgr_color': (0, 255, 0) # BGR (Blue=0, Green=255, Red=0)
    },
    {
        'name': 'Blue',
        'ranges': [
            (np.array([100, 150, 50]), np.array([140, 255, 255]))
        ],
        'bgr_color': (255, 0, 0) # BGR (Blue=255, Green=0, Red=0)
    }
]

print("Starting Color Detection for: Red, Green, Blue")
print("Press 'q' to exit the application.")

# Initialize the video capture object (0 is usually the default camera)
cap = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

try:
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to retrieve frame.")
            time.sleep(1)
            continue

        # 1. Convert the frame from BGR to HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Initialize a combined mask for displaying the overall detection
        combined_mask = np.zeros_like(hsv_frame[:,:,0])

        # Iterate through each defined color to detect objects of that color
        for color_data in color_ranges:
            color_name = color_data['name']
            bgr_color = color_data['bgr_color']
            
            # --- Generate Mask for Current Color ---
            current_color_mask = None
            
            # Handle single or multiple HSV ranges (necessary for Red)
            for hsv_min, hsv_max in color_data['ranges']:
                mask_segment = cv2.inRange(hsv_frame, hsv_min, hsv_max)
                
                if current_color_mask is None:
                    current_color_mask = mask_segment
                else:
                    current_color_mask = cv2.bitwise_or(current_color_mask, mask_segment)

            # --- Refine and Combine Mask ---
            # Refine the mask using morphological operations
            kernel = np.ones((5, 5), np.uint8)
            refined_mask = cv2.erode(current_color_mask, kernel, iterations=1)
            refined_mask = cv2.dilate(refined_mask, kernel, iterations=2)
            
            # Accumulate this color's mask into the combined mask for visual output
            combined_mask = cv2.bitwise_or(combined_mask, refined_mask)

            # 5. Find contours of the detected areas
            contours, _ = cv2.findContours(refined_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 6. Draw bounding boxes around the largest detected objects
            if contours:
                # Iterate through all detected contours
                for contour in contours:
                    if cv2.contourArea(contour) > 500: # Only draw if area is significant
                        # Get the bounding rectangle coordinates
                        x, y, w, h = cv2.boundingRect(contour)

                        # Draw a rectangle around the detected object in the original frame
                        cv2.rectangle(frame, (x, y), (x + w, y + h), bgr_color, 2)

                        # Add label text
                        cv2.putText(frame, color_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr_color, 2)


        # 4. Apply the combined mask to the original frame for a single visual output
        combined_detection_result = cv2.bitwise_and(frame, frame, mask=combined_mask)

        # 7. Display the original and the masked frames
        cv2.imshow('Original Frame (Press Q to Quit)', frame)
        cv2.imshow('Combined Color Detection', combined_detection_result)

        # Break the loop when the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"An unexpected error occurred: {e}")

finally:
    # Release the video capture object and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")