from keras import models
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

model = models.load_model("./my_model.h5", compile=False)

labels = open("labels.txt", "r", -1, "utf-8").readlines()

def predict(fpath):  
    image = Image.open(fpath).convert("RGB")
    size = (244, 244)
    image = ImageOps.fit(image, size)
    image_array = np.asarray(image).reshape((244, 244, 3))

    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    m_data = np.ndarray(shape=(1, 244, 244, 3), dtype=np.float32)

    m_data[0] = normalized_image_array
    res = model.predict(m_data)

    diagnose = labels[np.argmax(res[0])]
    
    print(res[0])
    print("Результат: {}".format(diagnose))

    return diagnose