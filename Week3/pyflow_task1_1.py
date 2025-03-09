import cv2
import numpy as np
import PyFlow

# Define image paths
img1_path = "C:/Users/Julia/Downloads/C6/ProjectC6/000045_10.png"
img2_path = "C:/Users/Julia/Downloads/C6/ProjectC6/000045_11.png"

# Load images in grayscale
img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

# Check if images are loaded correctly
if img1 is None or img2 is None:
    print("Error: Could not load images. Check file paths.")
    exit()

# Initialize Optical Flow method (Coarse2Fine)
of_method = PyFlow.Coarse2FineFlow()

# Compute Optical Flow
flow = of_method.estimate(img1, img2)

# Convert Optical Flow to an HSV image for visualization
hsv = np.zeros((img1.shape[0], img1.shape[1], 3), dtype=np.uint8)
hsv[..., 1] = 255
mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
hsv[..., 0] = ang * 180 / np.pi / 2
hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# Save the Optical Flow image
output_path = "C:/Users/Julia/Downloads/C6/optical_flow_result.png"
cv2.imwrite(output_path, flow_rgb)

# Display the result
cv2.imshow("Optical Flow", flow_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
