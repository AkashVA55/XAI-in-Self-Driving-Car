import cv2

# Load and preprocess the image
image = cv2.imread('C:/Users/AKASH V A/Downloads/simulator-windows-64/data/IMG/center_2024_05_09_00_13_13_911.jpg')  # Replace 'path_to_image.jpg' with the path to your image file
#image = cv2.GaussianBlur(image,  (3, 3), 0)
image = cv2.resize(image, (200,66))
  # Resize the image to match the model's input shape
image = image.astype('float32') / 255.0  # Normalize the pixel values
#image = np.expand_dims(image, axis=0)  # Add batch dimension

# Generate explanations for the chosen instance
image.shape


import lime
from lime import lime_image
from keras.models import load_model
# Initialize LIME explainer
explainer = lime_image.LimeImageExplainer()
model = load_model('C:/Users/AKASH V A/Downloads/model(1).h5')
# Generate explanations for the chosen instance
explanation = explainer.explain_instance(image, model.predict, top_labels=1, hide_color=0, num_samples=1000)


import matplotlib.pyplot as plt

from skimage.segmentation import mark_boundaries

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
img_boundry1 = mark_boundaries(temp / 2 + 0.5, mask)
plt.imshow(img_boundry1)
plt.show()