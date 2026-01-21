import glob

from PIL import Image
from lang_sam import LangSAM    

import os
import cv2
import numpy as np

from sklearn.utils import shuffle


def load_images_and_labels(folder_path):

    image_paths = sorted(glob.glob(os.path.join(folder_path, "*.png")))
    images = [cv2.imread(img_path) for img_path in image_paths]

    return images


def cv2_to_pil(image):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    return image


def draw_bb(image, bbox):

    top_left = (int(bbox[0]), int(bbox[1]))
    bottom_right = (int(bbox[2]), int(bbox[3]))

    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
    cv2.imshow("img_bbox", image)


def draw_mask(image, mask):

    mask = np.array(mask)
    mask = np.stack((mask, mask, mask), axis=-1)
    image = np.where(mask, (0, 255, 0), image).astype(np.uint8)
    cv2.imshow("img_mask", image)


def main():

    folder_path_train = "./data/"

    images_train = load_images_and_labels(folder_path_train)

    model = LangSAM()
    text_prompt = "tactile. contact."

    for image in images_train:
        
        cv2.imshow("img", image)

        img = cv2_to_pil(image)
        results = model.predict([img], [text_prompt])
        
        num_preds = results[0]['scores'].size

        if num_preds > 0:
            
            for i in range(num_preds):
                draw_bb(image, results[0]['boxes'][i])
                draw_mask(image, results[0]['masks'][i])

        if cv2.waitKey(0) & 0xFF == ord('s'):
            continue
        elif cv2.waitKey(0) & 0xFF == ord('q'):
            break
        


if __name__ == '__main__':
    main()