import cv2                                              # OpenCV modul : image processing
import time                                             # to measure code execution time
import os                                               # OperationSystem : libraries, files, paths
import joblib                                           # for loading the model
import numpy as np                                      # Numerical Python : calculatons
import matplotlib.pyplot as plt                         # showing modified image in an interactive window
from concurrent.futures import ThreadPoolExecutor       # for multithreaded working
from skimage.feature import local_binary_pattern        # lbp feature calculation



# starting the total process timer
total_time = time.perf_counter()

# paths
MODEL_PATH = r"\path\to\the\model\you\want\to\use"
IMAGE_PATH = r"\path\to\the\image\you\want\to\use"

# size of the patch (both side equal) and stepping size
PATCH_SIZE = 64
STRIDE = 64

# how many threads should the program use maximum, for me uses 12 out of 16
MAX_WORKERS = min(32, (os.cpu_count() or 1) + 4)

# every feature is scaled between 0 and 180
# Hue lookup tables.
_HUE_ANGLES = np.arange(180, dtype=np.float32) * (2.0 * np.pi / 180.0)
_HUE_SIN = np.sin(_HUE_ANGLES)
_HUE_COS = np.cos(_HUE_ANGLES)
_HUE_SCALE = 90.0 / np.pi

# Saturation and Value lookup tables
_SV_MEAN_SCALE = 180.0 / 255.0
_SV_STD_SCALE = 180.0 / 127.5



# how to get the hue features
def get_hue_features(im_hsv):

    # getting the hue original value and saturation value between 0-1 (for weights)
    hue = im_hsv[:, :, 0].astype(np.intp, copy=False)
    sat = im_hsv[:, :, 1].astype(np.float32, copy=False) / 255.0

    # using the saturation weight to calculate the sum of angles
    sin_sum = np.sum(sat * _HUE_SIN[hue], dtype=np.float32)
    cos_sum = np.sum(sat * _HUE_COS[hue], dtype=np.float32)
    sat_sum = np.sum(sat, dtype=np.float32)

    # return 0 for hue mean and 180 for stddev, when saturation is almost zero.
    if sat_sum <= 1e-6:
        return 0.0, 180.0

    # using arctan2 to keep quadrant information, 
    # getting the direction of hue in degree (using hue_scale table), 
    # using % 180 to warp the value back to 1 instead of 181
    hue_mean = (np.arctan2(sin_sum, cos_sum) * _HUE_SCALE) % 180.0

    # r is the normalized (by sat_sum) lenght of the vector (hue_mean), 
    # forcing it between 0 and 1 by clip() as log(0) is undefined and lenght of R should be 0-1 anyway
    r = np.sqrt(sin_sum * sin_sum + cos_sum * cos_sum) / sat_sum
    r = float(np.clip(r, 1e-6, 1.0))

    # if R is 1, the value is 0 (hues are identical), 
    # if hue is more diverse, R become less then 1, log(r) will become increasing negative value, the sttdev is increasing
    hue_std = np.sqrt(-2.0 * np.log(r)) * _HUE_SCALE

    return float(hue_mean), float(hue_std)



# how to get the saturation/value features
def get_saturation_value_features(im_hsv):

    # get the mean and standard deviation of the given HSV image
    mean, std = cv2.meanStdDev(im_hsv)

    # get the mean of channel 1 and 2 which is saturation and value, 
    # values are converted to a 180 scale
    sat_mean = mean[1, 0] * _SV_MEAN_SCALE
    val_mean = mean[2, 0] * _SV_MEAN_SCALE

    # get the stddev of channel 1 and 2 which is saturation and value, 
    # values are converted to a 180 scale
    sat_std = std[1, 0] * _SV_STD_SCALE
    val_std = std[2, 0] * _SV_STD_SCALE

    return sat_mean, val_mean, sat_std, val_std



# how to get the lbp histogram feature
def get_lbp_features(im_gray):

    # using grayscale version (only intrested in texture, not color), 
    # with 8 neighbor pixels, 
    # 1 radius,
    # uniform method are all choosen by trail and error 
    lbp = local_binary_pattern(im_gray, 8, 1, method="uniform")

    # histogram shows how frequantly a value occures, 
    # ravel turn the image values into a list, 
    # bin is the category number, 
    # range means 0-10, 
    # densit=true normalize the histogram, so wont be effected by image size
    hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)

    # normalizing the value to 180 scale
    hist = hist.astype(np.float32) * 180.0

    return hist



# to describe how to get the category and features
def get_category_feature_array(image_path, image_name):

    # to read the image
    im = cv2.imread(image_path)

    # to get the grayscale and HSV version of the image for LBP and HSV features
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    # to get the normalized mean and stddev of hue
    hue_mean, hue_std = get_hue_features(im_hsv)

    # to get the normalized mean and stddev of saturation and value
    sat_mean, val_mean, sat_std, val_std = get_saturation_value_features(im_hsv)

    # to get the normalized lbp histogram
    lbp_features = get_lbp_features(im_gray)

    # to make a color features array of float32 (required format for sklearn training is either float32 or float64)
    color_features = np.array([hue_mean, sat_mean, val_mean, hue_std, sat_std, val_std], dtype=np.float32)

    # the final feature array is made of color and lbp features combined
    feature = np.concatenate((color_features, lbp_features))

    return feature



def evaluate_image():

    # print that the model is being loaded
    print("Loading model...")

    # loads the trained model using the saved path
    model = joblib.load(MODEL_PATH)

    # print that the image is being loaded
    print("Loading image...")

    # this reads the image from the given image path
    image = cv2.imread(IMAGE_PATH)

    # checks if the image is loaded, if not raise a filenotfound error
    if image is None:
        raise FileNotFoundError(IMAGE_PATH)

    # save the first 2 element which is height and width of the image
    height, width = image.shape[:2]

    # prints out the image dimensions
    print(f"Image: {width} x {height}")

    # collect all the patches:
    patches = []
    positions = []

    # moves through the image vertically using the hight-patch as endpoint and stride as stepping distance (same as patch so no overlap or skips)
    for y in range(0, height - PATCH_SIZE + 1, STRIDE):
        # moves through the image horizontally using the width-patch as endpoint and stride as stepping distance (same as patch so no overlap or skips)
        for x in range(0, width - PATCH_SIZE + 1, STRIDE):
            # cuts out one patch,
            # starts from y and x pixels and ends with y+patch and x+patch pixels
            patches.append(image[y:(y + PATCH_SIZE), x:(x + PATCH_SIZE)])
            # save the position of the patch (by most left-top pixel position)
            positions.append((x, y))

    # prints the patches number
    print(f"Patches: {len(patches):,}")

    # shows that the feature extraction is starting
    print("Extracting features...")

    # start time for feature extraction
    feature_extraction_start = time.perf_counter()

    # multithreaded work
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # extracts features from every patch and stores it
        features = list(executor.map(extract_features, patches))

    # converting the extracted features into a float32 type numpy array (can be float32 or float64)
    features = np.asarray(features, dtype=np.float32)

    # prints how much time the feature extraction take (current - starting time)
    print(f"Extraction time: {time.perf_counter() - feature_extraction_start:.2f} seconds")

    # shows that the prediction starts
    print("Predicting...")

    # records the start time for prediction
    prediction_start = time.perf_counter()

    # useing the trained model to predict the class of every patch
    predictions = model.predict(features)

    # prints how long prediction took (current - starting time)
    print(f"Prediction time: {time.perf_counter() - prediction_start:.2f} seconds")

    # defining the color used for the classes (BGR format)
    CLOUD_COLOR = (0, 0, 150) 
    OTHER_COLOR   = (150, 0, 0)   
    SKY_COLOR = (0, 150, 0)

    # connecting each prediction number to a color (there is only 3 main coloring class: cloud/sky/other)
    colors = {0: SKY_COLOR, 1: CLOUD_COLOR, 2: OTHER_COLOR, 3: SKY_COLOR, 4: CLOUD_COLOR, 5: OTHER_COLOR, 6: SKY_COLOR, 7: CLOUD_COLOR, 8: OTHER_COLOR,}

    # saving (copying) the unmodified image, this will be the overlay layer
    overlay = image.copy()

    # loops through every patch position and its prediction
    for (x, y), prediction in zip(positions, predictions):
        # gets the color assigned to the current prediction
        color = colors.get(int(prediction), (255, 255, 255))

        # draws a filled rectangle over to the overlay image's current patch 
        # determined by the most top-left and most buttom-right pixels
        # using color dictionaries category-color pairs
        # -1 fills the whole patch
        cv2.rectangle(overlay, (x, y), (x + PATCH_SIZE, y + PATCH_SIZE), color, -1)

    # blends the original image and the colored overlay in 50-50 transparency
    result = cv2.addWeighted(image, 0.5, overlay, 0.5, 0)

    # creating a figure with the specified size
    plt.figure(figsize=(14, 8))

    # converts the image from bgr to rgb and displays it
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

    # hides the axis around the displayed image
    plt.axis("off")

    # adds a title which shows the number of evaluated patches
    plt.title(f"Evaluated image — {len(patches):,} patches")

    # prints the total time used for the complete extraction, prediction and showing the result
    print(f"total time: {time.perf_counter() - total_time:.2f} seconds")

    # displays the final image window
    plt.show()

# doing the full evalutation
evaluate_image()
