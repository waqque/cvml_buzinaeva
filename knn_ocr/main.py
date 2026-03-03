import cv2
import numpy as np
from pathlib import Path
from skimage.measure import regionprops, label
from skimage.io import imread

data_directory = Path("./task")

def extractor(image):
    if image.ndim == 2:
        binary = image
    else:
        gray = np.mean(image,2).astype('u1')
        binary = gray > 6  
    lb = label(binary)
    props = regionprops(lb)
    main_region = props[0]
    height, width = main_region.image.shape
    center_y, center_x = main_region.centroid_local
    center_y = center_y / height if height > 0 else 0
    center_x = center_x / width if width > 0 else 0

    return np.array([
        main_region.eccentricity,
        main_region.euler_number,
        main_region.extent,
        center_y,
        center_x
    ], dtype='f4')

def make_train(path):
    train = []
    responses = []
    class_counter = 0
    label_names = {}

    for class_subfolder in sorted(path.glob("*")):
        class_counter += 1
        label_name = class_subfolder.name

        if label_name.startswith('s') and len(label_name) > 1:
            label_name = label_name[1:]

        label_names[class_counter] = label_name
        print(class_subfolder.name, class_counter)

        for image_file in class_subfolder.glob("*.png"):
            img_data = imread(image_file)
            features = extractor(img_data)
            train.append(features)
            responses.append(class_counter)

    train = np.array(train, dtype='f4').reshape(-1, 5)
    responses = np.array(responses, dtype='f4').reshape(-1, 1)

    return train, responses, label_names

train_data, train_labels, label_dictionary = make_train(data_directory / "task/train")
knn_model = cv2.ml.KNearest.create()
knn_model.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)

recognition_results = []

for file_number in range(7):
    current_image = imread(data_directory / "task" / f"{file_number}.png")
    grayscale_img = np.mean(current_image, 2).astype('u1')
    binary_img = grayscale_img > 6 

    labeled_components = label(binary_img)
    component_regions = regionprops(labeled_components)
    component_regions = sorted(component_regions, key=lambda x: x.bbox[1])

    merged_boxes = []
    for region in component_regions:
        y0, x0, y1, x1 = region.bbox
        centroid_x = region.centroid[1]

        if merged_boxes and abs(centroid_x - merged_boxes[-1][4]) < 8:
            previous = merged_boxes[-1]
            merged_boxes[-1] = (
                min(y0, previous[0]), min(x0, previous[1]),
                max(y1, previous[2]), max(x1, previous[3]),
                (centroid_x + previous[4]) / 2
            )
        else:
            merged_boxes.append((y0, x0, y1, x1, centroid_x))

    recognized_text = ""
    previous_x = None

    for bounding_box in merged_boxes:
        y0, x0, y1, x1, _ = bounding_box

        if previous_x is not None and x0 - previous_x > 25:
            recognized_text += " "

        character_image = binary_img[y0:y1, x0:x1]
        char_features = extractor(character_image).reshape(1, 5)
        ret_val, predictions, neighbors, distances = knn_model.findNearest(char_features, 3)
        recognized_text += label_dictionary[int(predictions[0][0])]
        previous_x = x1

    print(recognized_text)
    recognition_results.append(recognized_text)

ground_truth = [
    'C is LOW-LEVEL', 'C++ is POWERFUL', 'Python is INTUITIVE',
    'Rust is SAFE', 'LUA is EASY', 'Javascript is UGLY', 'PHP sucks'
]

correct_predictions = 0
total_characters = sum(len(text) for text in ground_truth)

for idx in range(7):
    for char_idx in range(min(len(recognition_results[idx]), len(ground_truth[idx]))):
        if recognition_results[idx][char_idx] == ground_truth[idx][char_idx]:
            correct_predictions += 1


print(f'Точность распознавания: {correct_predictions/total_characters:.4f}')

