# Video Surveillance for Road Traffic Monitoring

## Project Overview
This project focuses on **video surveillance for road traffic monitoring**, using object detection and multi-camera tracking techniques. The main objective is to detect and track vehicles across multiple camera views while addressing challenges such as camera handover, re-identification (Re-ID), and projection errors.

## Team Members
- **Lucía Arribas**  
- **Laura Salort**  
- **Miren Samaniego**  
- **Júlia Bujosa**  

## Methodology
### 1. Object Detection
- A **pretrained model** was fine-tuned on the dataset to detect vehicles.
- The **COCOEvaluator** metric was used, achieving an **mAP of 86.33%**.
- Performance was best on sequences **S01 and S04**, as the object detector was trained on those.

### 2. Object Tracking (Single Camera)
- **Simple tracking methods** were initially used. (simple_tracking.py)
- Further improvement was attempted with **DeepSORT tracking**. (deep_sort_multi_track_ROI_crop.py)
- The model struggled with **S03** due to limited training data on that sequence.

### 3. Multi-Camera Tracking
- **Projection to GPS coordinates** was applied to match vehicle locations across cameras. (reidentification.py)
- **Challenges:**
  - IDF1 scores were lower when using multi-camera tracking.
  - Alignment issues caused incorrect associations.
  - Some vehicles were not detected due to failures in single-camera tracking.

### 4. Enhancing Multi-Camera Tracking
- **FastReID** was used for vehicle re-identification. (reidentification_2.py)
- Added **deep appearance embeddings** to distinguish vehicles based on visual cues.
- **Challenges:**
  - Some parked cars were wrongly detected and assigned incorrect IDs across cameras.
  - Poor threshold selection and projection errors led to misidentifications.

## Results and Findings
- The model performed well on **S01 and S04** but had difficulties generalizing to **S03**.
- Large differences in **GPS projections** affected performance.
- The **appearance-based multi-camera tracking** improved identification but introduced new errors.

## Challenges & Future Improvements
### Challenges
- False detections due to overfitting on **S01 and S04**.
- **Frame alignment issues** due to video transmission noise.
- Poor calibration led to significant differences in projected GPS coordinates.

### Future Improvements
- **Find better thresholds** for object tracking and re-identification.
- Improve **appearance models** for vehicle distinction.
- For **S04**, consider only cameras pointing at the same streets or intersections.

### Conclusion
Although the model achieves good results, it is not as robust as desired due to **projection errors and misalignment issues**. Further refinements in threshold selection, calibration, and model generalization are needed to enhance multi-camera tracking performance.

