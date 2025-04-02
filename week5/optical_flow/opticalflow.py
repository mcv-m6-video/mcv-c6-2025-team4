    
import cv2 as cv 
import numpy as np 
import os
from tqdm import tqdm
rootdir="/gpfs/home/lsalort/mcv-c6-2025-team4/Week5/SoccerNet/SN-BAS-2025-frames/398x224/england_efl/2019-2020/2019-10-01 - Middlesbrough - Preston North End"


output_folder = rootdir+"_flow/" 
image_folder=rootdir
os.makedirs(output_folder, exist_ok=True)

# Get sorted list of image files
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

# Read the first frame
first_frame = cv.imread(os.path.join(image_folder, image_files[0]))
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
# Create an image filled with zeros with the same dimensions as the frame
mask = np.zeros_like(first_frame)
mask[..., 1] = 255  # Set saturation to maximum
final_flow=[]
for i in tqdm(range(1, len(image_files))):
    frame = cv.imread(os.path.join(image_folder, image_files[i]))
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Calculate dense optical flow using Farneback method
    flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # final_flow.append(flow)


    # Update previous frame
    prev_gray = gray

    with open(output_folder+str(i)+'.npy', 'wb+') as f:
        np.save(f, np.array(flow))


