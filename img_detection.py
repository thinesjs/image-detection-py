import urllib.request
from io import BytesIO
import cv2
import numpy as np
from keras.preprocessing import image
# from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from PIL import Image
import requests
from matplotlib import pyplot as plt
from mtcnn import MTCNN

# marking faces dependencies
from matplotlib.patches import Rectangle

# comparing faces
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine

# important versions
# keras = 2.12
# tensorflow-macos = 2.13.0

# model = ResNet50(weights='imagenet')
# target_size = (224, 224)


def store_image(url, local_file_name):
    with urllib.request.urlopen(url) as resource:
        with open(local_file_name, 'wb') as f:
            f.write(resource.read())


def highlight_faces(image_path, faces):
    # display image
    img = plt.imread(image_path)
    plt.imshow(img)

    ax = plt.gca()

    # for each face, draw a rectangle based on coordinates
    for face in faces:
        x, y, width, height = face['box']
        face_border = Rectangle((x, y), width, height,
                                fill=False, color='red')
        ax.add_patch(face_border)
    plt.show()


def extract_face_from_image(image_path, required_size=(224, 224)):
    # load image and detect faces
    image = plt.imread(image_path)
    detector = MTCNN()
    faces = detector.detect_faces(image)

    # extract the bounding box from the requested face
    x1, y1, width, height = faces[0]['box']
    x2, y2 = x1 + width, y1 + height

    # extract the face
    face_boundary = image[y1:y2, x1:x2]

    image = cv2.resize(face_boundary, required_size)

    return image


# # Display the first face from the extracted faces
# plt.imshow(extracted_face[0])
# plt.show()

def get_model_scores(faces):
    samples = np.asarray(faces, 'float32')

    # prepare the data for the model
    samples = preprocess_input(samples, version=2)

    # create a vggface model object
    model = VGGFace(
        model='resnet50',
        include_top=False,
        input_shape=(224, 224, 3),
        pooling='avg'
    )

    # perform prediction
    return model.predict(samples)

# def predict(model, img, target_size, top_n=3):
#     """
#   Args:
#     model: keras model
#     img: PIL format image
#     target_size: (width, height) tuple
#     top_n: # of top predictions to return
#   Returns:
#     list of predicted labels and their probabilities
#   """
#     if img.size != target_size:
#         img = img.resize(target_size)
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     predictions = model.predict(x)
#     return decode_predictions(predictions, top=top_n)[0]
# img = Image.open() # local image
# img = Image.open(BytesIO(requests.get(image3).content))  # network image
# print(predict(model, img, target_size))


if __name__ == "__main__":
    # network sources for test images
    image1 = "https://i.imgur.com/wpxMwsR.jpeg"  # (image of elephant)
    image2 = "https://www.janek.com/wp-content/uploads/2020/01/Sales-Managers-Beware-the-Numbers-Trap.jpg"  # (image of farmer)
    image3 = "https://d37plr7tnxt7lb.cloudfront.net/156.jpg"  # (image of param)
    image4 = "https://assets.ey.com/content/dam/ey-sites/ey-com/en_my/topics/accelerating-growth/ey-parmjit.jpg"  # (another image of param)
    image5 = "https://apicms.thestar.com.my/uploads/images/2020/05/06/670856.jpg" # (not an image of parmjit)

    # store the image from network src
    store_image(image3, "param.jpg")
    store_image(image4, "param1.jpg")
    store_image(image5, "notparam.jpg")

    # read the image
    image = plt.imread('param.jpg')
    similarImage = plt.imread('param1.jpg')

    # using mtcnn model to identify faces in the image
    detector = MTCNN()

    # faces = detector.detect_faces(similarImage)
    # to print the array with key points
    # for face in faces:
    #     print(face)

    # use defined function to highlight possible faces with a rectangular box
    faces = detector.detect_faces(plt.imread('param1.jpg'))
    highlight_faces('param1.jpg', faces)
    faces = detector.detect_faces(plt.imread('param.jpg'))
    highlight_faces('param.jpg', faces)

    # extract faces from images and into array to compare the model scores
    faces = [extract_face_from_image(image_path)
             for image_path in ['param1.jpg', 'param.jpg']]

    model_scores = get_model_scores(faces)

    if cosine(model_scores[0], model_scores[1]) <= 0.4:
        print("faces match")
    else:
        print("faces do not match")
