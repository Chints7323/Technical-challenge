import numpy as np
import cv2
from ultralytics import YOLO
import pandas as pd
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Object detection and change detection between base and test images.")
    parser.add_argument('base_file', type=str, help="Path to the base .npz file.")
    parser.add_argument('test_file', type=str, help="Path to the test .npz file.")
    parser.add_argument('--custom_model', type=str, default="models/custom_yolo.pt", help="Path to the custom-trained YOLO model.")
    parser.add_argument('--pretrained_model', type=str, default="models/yolov8n.pt", help="Path to the pre-trained YOLO model.")
    return parser.parse_args()

# Function to calculate Intersection over Union (IoU)
# IoU is a measure of the overlap between two bounding boxes and is used to determine how similar they are.
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area != 0 else 0

# Function to perform Non-Maximum Suppression (NMS) to filter redundant boxes
# NMS helps in removing multiple detections of the same object by keeping only the best one.
def non_maximum_suppression(detections, iou_threshold=0.5):
    if len(detections) == 0:
        return detections

    boxes = detections[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
    scores = detections['confidence'].values.tolist()
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=iou_threshold)

    return detections.iloc[indices.flatten()]

# Function to compare detections between base and test images considering labels
# This function identifies objects that have appeared or disappeared between the base and test images.
def compare_detections(base_detections, test_detections, iou_threshold=0.5):
    appeared = []
    disappeared = []

    for _, test_det in test_detections.iterrows():
        found = False
        for _, base_det in base_detections.iterrows():
            iou = calculate_iou(test_det[['xmin', 'ymin', 'xmax', 'ymax']].values, base_det[['xmin', 'ymin', 'xmax', 'ymax']].values)
            if iou > iou_threshold and test_det['name'].lower() == base_det['name'].lower():
                found = True
                break
        if not found:
            appeared.append(test_det)

    for _, base_det in base_detections.iterrows():
        found = False
        for _, test_det in test_detections.iterrows():
            iou = calculate_iou(base_det[['xmin', 'ymin', 'xmax', 'ymax']].values, test_det[['xmin', 'ymin', 'xmax', 'ymax']].values)
            if iou > iou_threshold and base_det['name'].lower() == test_det['name'].lower():
                found = True
                break
        if not found:
            disappeared.append(base_det)

    return pd.DataFrame(appeared), pd.DataFrame(disappeared)

# Function to run object detection and return detections
# This function uses a given YOLO model to perform object detection on an image and returns the detections.
def run_detection(model, image):
    results = model.predict(image, verbose=False)  # Disable verbose logs
    detections = pd.DataFrame(results[0].boxes.xyxy.cpu().numpy(), columns=['xmin', 'ymin', 'xmax', 'ymax'])
    detections['confidence'] = results[0].boxes.conf.cpu().numpy()
    detections['class'] = results[0].boxes.cls.cpu().numpy()
    detections['name'] = [results[0].names[int(cls)].lower() for cls in results[0].boxes.cls.cpu().numpy()]  # Convert to lowercase
    return detections

# Function to display processing window
# This function shows a window indicating the progress of processing base images.
def display_processing_window(percentage):
    processing_image = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.putText(processing_image, "Processing Base Images...", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(processing_image, f"{percentage:.2f}% complete", (130, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow('Processing', processing_image)
    cv2.waitKey(5)  # Reduced wait time to 5 milliseconds

# Function to display navigation bar with controls
# This function adds a navigation bar with playback instructions at the bottom of the image.
def display_navigation_bar(image):
    nav_image = np.zeros((50, image.shape[1], 3), dtype=np.uint8) + 255  # White background
    instructions = "Press 'p' to Pause/Play, 'q' or 'ESC' to quit"
    cv2.putText(nav_image, instructions, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    combined_image = np.vstack((image, nav_image))
    return combined_image

def process_base_images(base_images, custom_model):
    base_detections_dict = {}
    base_classes = set()
    num_base_images = base_images.shape[0]
    for i in range(num_base_images):
        image = base_images[i]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        custom_detections = run_detection(custom_model, image_rgb)
        base_detections_dict[i] = non_maximum_suppression(custom_detections).reset_index(drop=True)
        base_classes.update(base_detections_dict[i]['name'].unique())  # Collect unique class names

        # Update processing window
        percentage_complete = (i + 1) / num_base_images * 100
        display_processing_window(percentage_complete)
    
    # Close processing window
    cv2.destroyWindow('Processing')
    
    return base_detections_dict, base_classes

def process_test_images(test_images, base_detections_dict, base_classes, custom_model, pretrained_model, test_gpsvs, test_compassvs, canvas):
    paused = False
    for i in range(test_images.shape[0]):
        canvas[:, :, :] = 0
        image, gps, compass = test_images[i], test_gpsvs[i], test_compassvs[i]

        if image.shape[2] == 4:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        else:
            image_rgb = image

        custom_detections = run_detection(custom_model, image_rgb)
        custom_detections = non_maximum_suppression(custom_detections)
        custom_detections['pretrained'] = False

        pretrained_detections = run_detection(pretrained_model, image_rgb)
        pretrained_detections = non_maximum_suppression(pretrained_detections)
        pretrained_detections['pretrained'] = True

        combined_detections = pd.concat([custom_detections, pretrained_detections]).reset_index(drop=True)

        final_detections = []
        for _, current_det in combined_detections.iterrows():
            keep = True
            for fd in final_detections:
                iou = calculate_iou(current_det[['xmin', 'ymin', 'xmax', 'ymax']].values, fd[['xmin', 'ymin', 'xmax', 'ymax']].values)
                if iou > 0.5:
                    keep = False
                    break
            if keep:
                final_detections.append(current_det)

        final_detections = pd.DataFrame(final_detections)

        if 'pretrained' in final_detections.columns:
            unknown_detections = final_detections[final_detections['pretrained'] == True]
            known_detections = final_detections[final_detections['pretrained'] == False]
        else:
            unknown_detections = pd.DataFrame(columns=final_detections.columns)
            known_detections = final_detections

        appeared, disappeared = compare_detections(base_detections_dict.get(i, pd.DataFrame(columns=known_detections.columns)), known_detections)
        appeared['pretrained'] = False
        disappeared['pretrained'] = False

        for idx, unknown_det in unknown_detections.iterrows():
            if unknown_det['name'] in base_classes:
                unknown_det['pretrained'] = False
                appeared = pd.concat([appeared, unknown_det.to_frame().T], ignore_index=True)
                unknown_detections.drop(idx, inplace=True)

        def display_detections(image, detections, color, label_suffix='', thickness=1, font_scale=0.3, alpha=1.0):
            for _, row in detections.iterrows():
                label = row['name']
                if row['pretrained']:
                    label = f"Unknown: {row['name']}"
                elif label_suffix:
                    label = f"{label_suffix}: {label}"
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

                overlay = image.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
                cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

                label_y = y1 - 2 if y1 - 2 > 10 else y1 + 20
                label_x = x1 if x1 + 100 < image.shape[1] else x1 - 100

                if label_y < 0:
                    label_y = y2 + 20

                if label_y > image.shape[0] - 10:  # Prevent label from going out of the image
                    label_y = y1 - 2 if y1 - 2 > 10 else y1 + 20

                cv2.putText(image, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, int(thickness * alpha))

        final_image = image_rgb.copy()

        # Draw unknown objects (detected by pre-trained model)
        display_detections(final_image, unknown_detections, (255, 0, 0), label_suffix='Unknown')

        # Draw appeared objects
        display_detections(final_image, appeared, (0, 255, 0), label_suffix='Appeared', thickness=1, font_scale=0.4)

        # Draw disappeared objects 
        display_detections(final_image, disappeared, (0, 0, 255), label_suffix='Disappeared', thickness=1, font_scale=0.3, alpha=0.3)

        combined_image = display_navigation_bar(final_image)

        cv2.imshow('Test', combined_image)

        x_in_map = int(gps[0] * 150) + canvas.shape[1] // 2
        y_in_map = canvas.shape[0] // 2 - int(gps[1] * 150) - canvas.shape[0] // 4
        cv2.circle(canvas, (x_in_map, y_in_map), 12, (0, 0, 255), 2)
        angle = np.arctan2(compass[1], compass[0]) - np.pi / 2
        nx_in_map = x_in_map + int(18 * np.cos(angle))
        ny_in_map = y_in_map + int(18 * np.sin(angle))
        cv2.line(canvas, (x_in_map, y_in_map), (nx_in_map, ny_in_map), (0, 255, 0), 1)
        cv2.imshow('map', canvas)

        k = cv2.waitKey(10)

        if k == ord('p'):
            paused = not paused
        elif k == 113 or k == 27:
            break

        while paused:
            k = cv2.waitKey(10)
            if k == ord('p'):
                paused = not paused
            elif k == 113 or k == 27:
                break

    cv2.destroyAllWindows()

def main():
    args = parse_arguments()

    custom_model = YOLO(args.custom_model)  # Load custom-trained YOLO model
    pretrained_model = YOLO(args.pretrained_model)  # Load pre-trained YOLO model

    # Load the provided .npz files
    loaded_base = np.load(args.base_file)
    loaded_test = np.load(args.test_file)

    base_images = loaded_base["images"]
    test_images = loaded_test["images"]
    base_gpsvs = loaded_base["gps"]
    test_gpsvs = loaded_test["gps"]
    base_compassvs = loaded_base["compass"]
    test_compassvs = loaded_test["compass"]
    canvas = np.zeros((800, 800, 3))

    # Process base images
    base_detections_dict, base_classes = process_base_images(base_images, custom_model)

    # Process test images and compare with base detections
    process_test_images(test_images, base_detections_dict, base_classes, custom_model, pretrained_model, test_gpsvs, test_compassvs, canvas)

if __name__ == "__main__":
    main()
