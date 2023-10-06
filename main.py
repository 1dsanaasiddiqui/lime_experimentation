import numpy as np
import matplotlib.pyplot as plt
from skimage.color import gray2rgb
from sklearn.datasets import fetch_openml
from skimage.color import label2rgb, rgb2gray  # Import rgb2gray


mnist = fetch_openml('mnist_784')

# Make each image color so lime_image works correctly
X_vec = np.stack([gray2rgb(iimg) for iimg in mnist['data'].values.reshape((-1, 28, 28))], 0).astype(np.uint8)
y_vec = mnist.target.astype(np.uint8)
# #%matplotlib inline
fig, ax1 = plt.subplots(1,1)
ax1.imshow(X_vec[0], interpolation = 'none')
ax1.set_title('Digit: {}'.format(y_vec[0]))
plt.savefig('digit_image.png')

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer

class PipeStep(object):
    """
    Wrapper for turning functions into pipeline transforms (no-fitting)
    """
    def __init__(self, step_func):
        self._step_func=step_func
    def fit(self,*args):
        return self
    def transform(self,X):
        return self._step_func(X)
makegray_step = PipeStep(lambda img_list: [rgb2gray(img) for img in img_list])
flatten_step = PipeStep(lambda img_list: [img.ravel() for img in img_list])
simple_rf_pipeline = Pipeline([
    ('Make Gray', makegray_step),
    ('Flatten Image', flatten_step),
    #('Normalize', Normalizer()),
    #('PCA', PCA(16)),
    ('RF', RandomForestClassifier())
                              ])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_vec, y_vec,
                                                    train_size=0.55)
simple_rf_pipeline.fit(X_train, y_train)
import os,sys
try:
    import lime
except:
    sys.path.append(os.path.join('..', '..')) # add the current directory
    import lime

from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
explainer = lime_image.LimeImageExplainer(verbose = False)
segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)

explanation = explainer.explain_instance(X_test[0], 
                                         classifier_fn = simple_rf_pipeline.predict_proba, 
                                         top_labels=10, hide_color=0, num_samples=10000, segmentation_fn=segmenter)

ax1.imshow(X_test[0], interpolation='none')
ax1.set_title('Digit being Explained: Digit {}'.format(y_test[0]))
plt.savefig('explanation_figure_digit_{}.png'.format(y_test[0]))




from lime import lime_image
from skimage.color import label2rgb
import matplotlib.pyplot as plt

# Assuming 'explanation' is an instance of LimeImageExplainer, and y_test[0] is a label
temp, mask = explanation.get_image_and_mask(y_test[0], positive_only=True, num_features=10, hide_rest=False, min_weight=0.01)

# Create the first subplot
fig, ax1 = plt.subplots(1, 2, figsize=(8, 4))
ax1[0].imshow(label2rgb(mask, temp, bg_label=0), interpolation='nearest')
ax1[0].set_title('Positive Regions for {}'.format(y_test[0]))

# Save the first subplot
plt.savefig('positive_regions_{}.png'.format(y_test[0]))

temp, mask = explanation.get_image_and_mask(y_test[0], positive_only=False, num_features=10, hide_rest=False, min_weight=0.01)

# Create the second subplot
ax1[1].imshow(label2rgb(3 - mask, temp, bg_label=0), interpolation='nearest')
ax1[1].set_title('Positive/Negative Regions for {}'.format(y_test[0]))

# Save the second subplot
plt.savefig('positive_negative_regions_{}.png'.format(y_test[0]))

# Display the plots (optional)
#plt.show()
