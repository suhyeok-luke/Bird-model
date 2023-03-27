import tensorflow as tf
import numpy as np
from keras.models import load_model
import pandas as pd
import json
from collections import OrderedDict

detector = tf.saved_model.load("./detector").signatures['default']
classifier = load_model('bird_classifier.h5')

df = pd.read_csv('species.csv')
species = df['Eng_name']

def run_detector(detector, path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    converted_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    result = detector(converted_img)
    result = {key:value.numpy() for key, value in result.items()}

    birdbox = []
    for i in range(len(result["detection_boxes"])):
        if result["detection_class_entities"][i].decode("ascii") == "Bird" and result["detection_scores"][i] >= 0.1 :
            ymin, xmin, ymax, xmax = tuple(result["detection_boxes"][i])
            im_height, im_width, c = img.shape
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
            birdbox.append(img[int(top):int(bottom), int(left):int(right)])

    return birdbox

def run_classifier(classifier, birdbox):
    result = []
    for bird in birdbox:
        input = tf.image.resize(bird, [150, 150])
        input = np.array(input/np.amax(input)*255, np.int32)

        y = classifier.predict(input[None])
        result.append(species[np.argmax(y, axis=1)[0]])

    return list(set(result))

def make_json(result):
    global species

    output_data = OrderedDict()
    output_data["species"] = result

    with open('./output/output.json', 'w', encoding="utf-8") as make_file:
        json.dump(output_data, make_file, ensure_ascii=False, indent="\t")

image_path = "./predict/predict(seagull).jpg" # temp
result = run_classifier(classifier, run_detector(detector, image_path))
make_json(result)
print(result)
