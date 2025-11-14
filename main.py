import cv2
import numpy as np
import time


color_ranges = [
    {
        'name': 'Red',
        'ranges': [
            (np.array([0, 100, 100]), np.array([10, 255, 255])),
            (np.array([160, 100, 100]), np.array([179, 255, 255])) 
        ],
        'bgr_color': (0, 0, 255) 
    },
    {
        'name': 'Green',
        'ranges': [
            (np.array([40, 70, 70]), np.array([80, 255, 255]))
        ],
        'bgr_color': (0, 255, 0) 
    },
    {
        'name': 'Blue',
        'ranges': [
            (np.array([100, 150, 50]), np.array([140, 255, 255]))
        ],
        'bgr_color': (255, 0, 0) 
    }
]

print("Starting Color Detection for: Red, Green, Blue")
print("Press 'q' to exit the application.")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

try:
    while True:
        #
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to retrieve frame.")
            time.sleep(1)
            continue

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        combined_mask = np.zeros_like(hsv_frame[:,:,0])
        for color_data in color_ranges:
            color_name = color_data['name']
            bgr_color = color_data['bgr_color']
            
      
            current_color_mask = None
            
            for hsv_min, hsv_max in color_data['ranges']:
                mask_segment = cv2.inRange(hsv_frame, hsv_min, hsv_max)
                
                if current_color_mask is None:
                    current_color_mask = mask_segment
                else:
                    current_color_mask = cv2.bitwise_or(current_color_mask, mask_segment)

         
            kernel = np.ones((5, 5), np.uint8)
            refined_mask = cv2.erode(current_color_mask, kernel, iterations=1)
            refined_mask = cv2.dilate(refined_mask, kernel, iterations=2)
            
        
            combined_mask = cv2.bitwise_or(combined_mask, refined_mask)

         
            contours, _ = cv2.findContours(refined_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            #
            if contours:
                
                for contour in contours:
                    if cv2.contourArea(contour) > 500: 
                        x, y, w, h = cv2.boundingRect(contour)

                       
                        cv2.rectangle(frame, (x, y), (x + w, y + h), bgr_color, 2)

                        
                        cv2.putText(frame, color_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr_color, 2)


        
        combined_detection_result = cv2.bitwise_and(frame, frame, mask=combined_mask)

       
        cv2.imshow('Original Frame (Press Q to Quit)', frame)
        cv2.imshow('Combined Color Detection', combined_detection_result)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"An unexpected error occurred: {e}")

finally:
    
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")