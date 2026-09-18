
<p align="center">
  <!-- APEX logo -->
  <img width="684" height="380" alt="apex 2 0" src="https://github.com/user-attachments/assets/8bddc042-70ac-4ffc-bba3-6ea98b67b6c9" />
</p>


<h1 align="center">APEX 2.0</h1>

<p align="center">
Atmospheric Pattern EXtractor recognises sky, cloud and other parts under day, night and dimness conditions.
</p>

---

## 1. Intro

APEX reads an image, cuts it into `64x64` patches, extracts features from each patch, and predicts one category for each patch.

The categories are made from two parts:

- light condition: `day`, `night`, `dimness`
- sky element: `sky`, `cloud`, `other`

That gives 9 categories total.

This repository is the rework of my thesis. The original version was made on a very weak notebook, under a time constraint, and it was the first time I made this type of project. At the end of it I recognised weaknesses in the data, the model, the speed and the hardware. APEX 2.0 corrects those weaknesses.

### Main changes

| Part | APEX 1.0 (thesis) | APEX 2.0 |
|---|---|---|
| Coverage | images are taken from 1 or 2 window's fixed postinion, where each element of the sky was at known location | every season, 5-6 countries, varying degree of sky coverage, sea level to above clouds, sunrise to late midnight |
| Sensor | cheap fixed webcam from the 2000s | flagship Sony and Samsung smartphone camera sensors |
| Resolution | 1080p | 48MP-50MP images, higher than 8K UHD (which is around 33MP), roughly 23x increase in pixels |
| Model | SVM, sufficient for the simple task | HGBDT based model, training data weights, changed distribution of train/test data. Required for the more complex task|
| Test accuracy | 96% accuracy, very high, but much more simple task | 92.9% accuracy for the much more complex task |
| Feature extraction | more than 20 minutes | 42.6 seconds with multithreaded feature extraction |
| Training time | 1 hour to 1.5 hours | 5 minutes 24 seconds |
| CPU | i5 5200U notebook CPU, 2 core 4 threads | desktop R7 5700X CPU, 8 core 16 threads |
| CPU measured | baseline | single core 165% better performance, multi core 585% better, 40% lower memory latency |

---

## 2. Requirements

The project was made with Python `3.10.11`.

Package versions are fixed, because there could be compatibility issues using different ones, also the training and testing results were made with these versions.

| Package | Version | Used for |
|---|---:|---|
| `pip` | `24.3.1` | package installation |
| `setuptools` | `75.6.0` | build/package support |
| `wheel` | `0.45.1` | wheel installation support |
| `numpy` | `1.23.5` | calculations, arrays, feature handling |
| `opencv-python` | `4.10.0.82` | image processing |
| `scikit-image` | `0.25.2` | LBP feature calculation |
| `matplotlib` | `3.10.9` | showing modified image in an interactive window |
| `scikit-learn` | `1.7.2` | model, training, parameter optimization, metrics |
| `joblib` | `1.6.0` | saving/loading the model, only needed if scikit-learn installs a non compatible version |

The `requirements.ipynb` notebook also has a version check cell to print the installed versions of the main packages.

---

## 3. Train

`train_apex_HGBDT.ipynb` prepares the training data, extracts the features, trains the model and saves it.

It does not train directly on full images. It first makes patches, because the final model also works patch by patch.

### 3.1 Image cutting

The cutting step cuts full masked images into smaller square patches.

Why:

- full images contain too much information at once
- the mask removes non-category specific parts by fully black areas, those areas should not become training samples
- training and prediction both work on `64x64` patches

How:

- the image is loaded from the edited image folder
- median blur with kernel size 9 is used to lower noise while keeping details
- the image is divided into `64x64` squares
- a square is saved only if it does not contain fully black `0/0/0` HSV pixels
- the saved patch name keeps the original image name and adds a cut id

Example patch name:

```text
day-cloud-1-17.jpg
```
### 3.2 Category extraction

The category comes from the image file name.

Why:

- the file name already contains the label
- train and test data stay simple to organize

How:

The name is split by `-` into:

- `timeofday`: day, night, dimness
- `partofsky`: sky, cloud, other

Categories:

| Category | Time of day | Sky element |
|---:|---|---|
| 0 | day | sky |
| 1 | day | cloud |
| 2 | day | other |
| 3 | night | sky |
| 4 | night | cloud |
| 5 | night | other |
| 6 | dimness | sky |
| 7 | dimness | cloud |
| 8 | dimness | other |


> Dimness means sunrise or dusk.

Example for categorization:

```text
day-cloud-1-17.jpg >> time of day = day, sky element = cloud >> category = 1
```
### 3.3 Feature extraction

Each patch is converted into a 16 dimensional feature array.

Why:

- the model cannot train directly on raw pixels
- color and texture both matter for sky/cloud/other separation

> Every feature is scaled between 0 and 180, which is a remnant part of the SVM code, has no effect for the HGBDT

Feature groups:

| Features | Count | What they describe |
|---|---:|---|
| hue mean, hue stddev | 2 | dominant hue direction and hue spread |
| saturation mean, saturation stddev | 2 | color strength |
| value mean, value stddev | 2 | brightness |
| LBP histogram | 10 | local texture |

Hue is handled as an angle, because hue wraps around. For that reason the hue mean and hue spread are calculated with circular statistics instead of a normal average.

LBP uses the grayscale image because it describes texture, not color. The histogram uses `density=True`, so it is normalized and not affected by image size.

Feature extraction is multithreaded.

Why:

- the first HGBDT extraction was still in progress for more than 30 minutes and was stopped because it took too much time
- with multithreading it takes 42.6 seconds

Thread count:

```python
max_workers = min(32, (os.cpu_count() or 1) + 4)
```

On the current desktop R7 5700X this uses 12 out of 16 threads.

### 3.4 Model training

The original SVM was insufficient to the more complex task. APEX 2.0 uses `HistGradientBoostingClassifier`.

Why HGBDT:

- it handles complex feature relations better
- it can handle multiple classes
- it worked much better than the original linear SVM model

Training uses:

- class weights, because the train category distribution is imbalanced
- 5 fold stratified cross validation, which tries to keep the original category ratio for the split
- `f1_macro` scoring, because macro F1 treats each class equally and is more useful for imbalanced data
- early stopping, so training stops when validation result stops improving
- saved final model with `joblib`

Main model settings:

| Setting | Value | Reason |
|---|---:|---|
| `max_iter` | 300 | maximum number of iterations, the total trees number are `max_iter * classes` |
| `early_stopping` | True | stop when there is no improvement |
| `validation_fraction` | 0.1 | 10% of training data used for early stopping validation |
| `n_iter_no_change` | 20 | stop if no improvement for 20 iterations |
| `class_weight` | custom dict | compensates imbalanced train category distribution |

Final hyperparameters:

| Parameter | Value | Reason |
|---|---:|---|
| `min_samples_leaf` | 30 | avoids decisions which results a less than 3 sample leaf, those are with low sample are often noise, like a 1/200 leaf node |
| `max_depth` | 16 | allows complex relations the deeper it can go, but more depth can make trees overfit |
| `learning_rate` | 0.06 | faster learning rate means less time to train but can overfit more easily, slower learn rate results a more robust model, but will take longer time to train, early stopping helps to reduce it |
| `l2_regularization` | 1 | reduction on leaf values to make the model less prone to overfitting , to large reduction will make the model underfit|

The `train_apex_HGBDT.ipynb` notebook will print out the:

- best parameters
- best cross validation macro F1
- test classification report
- test macro F1
- test balanced accuracy
- CV/test gap

Then it saves the model as `.joblib`.

---

## 4. Use

`use_apex_HGBDT.ipynb` loads a saved model and evaluates one full image.

The usage pipeline is:

1. load model
2. load image
3. cut the image into `64x64` patches
4. extract the same 16 features from every patch
5. predict the category of every patch
6. draw a colored overlay back onto the image
7. show the result and print timings

The patch size and stride are both 64:

```python
PATCH_SIZE = 64
STRIDE = 64
```

That means no overlap and no skipped patch area, except the last incomplete row/column if the image size is not divisible by 64. Only full patches are evaluated.

The feature functions must be the same as in training. The model expects the same feature order and the same 0-180 scaling.
> the scaling part is still a remnant part of the SVM code, has to use the same range as in training

Overlay colors:

| Element | BGR color |
|---|---|
| sky | `(0, 150, 0)` |
| cloud | `(0, 0, 150)` |
| other | `(150, 0, 0)` |

There are 9 predicted categories, but only 3 main coloring classes: sky, cloud and other.

The final image is made by blending the original image and the colored overlay with 50-50 transparency.

The `use_apex_HGBDT.ipynb` notebook will print out the:

- image size
- number of patches
- feature extraction time
- prediction time
- total time

Than shows the evaluated image in an interactive window.

---

## 5. Results

### 5.1 Original image 1 / APEX 1.0 / APEX 2.0
> the class-color pairs are:
> ![Sky](https://img.shields.io/badge/Sky-Blue-blue)
>![Clouds](https://img.shields.io/badge/Clouds-Yellow-yellow)
>![Other](https://img.shields.io/badge/Other-Dark_Pink-FF1493)   
<p align="center">
  <img width="841" height="210" alt="original_v_apex1 0_v_apex2 0" src="https://github.com/user-attachments/assets/cad010f0-9fba-4b97-9495-4287719a9073" />
</p>

This comparison reflects on the original image, the APEX 1.0 and the APEX 2.0 capabilities. 
- The first model is limited by fixed window views, weak webcam low resolution sensor and the original SVM classifier. The vast majority of the `other` part on the images is classified as `sky`, while the `sky` is classified as `cloud` at some parts. The `clouds` rough outline is visible.
- APEX 2.0 uses more diverse data, higher quality and resolution images and HGBDT classifier. That resulted a correctly identified `sky` part, and while all `clouds` were correctly identified, the model's conservative tendency caused some parts to be misclassified. The `other` part is mostly correctly evaluated, but at some parts, like the edge of the `cloud` is misclassified as that part don't have much training data
> Reasoning: having a half mask, half `cloud` edge containing patch or any other patch with 1 or more masked pixels are ignored and not created as training data.

---

### 5.2 Original image 2 / APEX 2.0
> the class-color pairs are:
> ![Sky](https://img.shields.io/badge/Sky-Green_overlay-green)
>![Clouds](https://img.shields.io/badge/Clouds-Red_overlay-red)
>![Other](https://img.shields.io/badge/Other-Blue_overlay-blue) 
<p align="center">
  <img width="2231" height="840" alt="original_v_apex2 0" src="https://github.com/user-attachments/assets/bc479780-5df7-4beb-9223-c5f78e803b5d" />
</p>

Another comparison is demonstrating that the more complex model has the ability to differentiate classes unrelated to position, unlike the first version. The `sky` part is correctly identified, and so does the `other` and `clouds`. The interesting part is that the `clouds` where mostly below the height of the mountain, but it don't mattered to the model, correctly recognized it above, the same and below mountain level.

---

### 5.3 Original image 3 / APEX 2.0  
> the class-color pairs are:
> ![Sky](https://img.shields.io/badge/Sky-Green_overlay-green)
>![Clouds](https://img.shields.io/badge/Clouds-Red_overlay-red)
>![Other](https://img.shields.io/badge/Other-Blue_overlay-blue) 
<p align="center">
  <img width="918" height="609" alt="original_v_apex2 0_2" src="https://github.com/user-attachments/assets/58c0bc09-1edd-48d1-9853-fb39f501d6c3" />

</p>

The last comparison shows a very difficult classification problem. Image took at 10km height, has lower positioned high density `clouds` and higher low density ones. The model correctly recognize the wing of the plane as `other` and the higher density `clouds` most part (wrongly assumes the dark spot inside it as `other`). Also APEX 2.0 successfully recognizes, that the sun is not a bright `cloud` or `sky` part, but something in the `other` category. The sun surrounding is correctly recognized as `clouds` and so does the `sky` part below it. At higher parts the model struggles and misclassify the `clouds` to `sky` and `other` class.

## 6. Future Improvements

- Add more images taken at night, including different levels of cloud coverage.
- Remove the 0–180 scaling.
- Experiment with more and better feature sets.
- If the model is robust enough, use the class ratio as a statistical measure.
- Experiment with deploying the model on embedded systems and evaluate its performance.
- Test the model with different types of cameras.
- Apply the model to video data.
- Estimate wind direction by analyzing the direction in which cloud regions appear and disappear in the video.
- Add other useful features to the `use_apex_HGBDT.ipynb` notebook.

