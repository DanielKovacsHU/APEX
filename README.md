<p align="center">
  <!-- APEX logo -->
  <img src="docs/assets/apex_logo.png" width="320" alt="APEX logo">
</p>

<h1 align="center">APEX 2.0</h1>

<p align="center">
Atmospheric Pattern EXtraction recognises sky, cloud and other parts under day, night and dimness conditions.
</p>

---

## 1. Intro

APEX reads an image, cuts it into `64x64` patches, extracts features from each patch, and predicts one category for each patch.

The categories are made from two parts:

- light condition: `day`, `night`, `dimness`
- sky element: `sky`, `cloud`, `other`

That gives 9 categories total.

This repository is the rework of my thesis. The original thesis was made on a very weak notebook, under a time constraint, and it was the first time I made this type of project. At the end of it I recognised weaknesses in the data, the model, the speed and the hardware. APEX 2.0 corrects those weaknesses.

### Main changes

| Part | APEX 1.0 (thesis) | APEX 2.0 |
|---|---|---|
| Coverage | images taken from 1 window, maximum 2, where each element of the sky was at fixed location | every season, 5-6 countries, multiple elements, varying degree of sky coverage, sea level to above clouds, sunrise to late midnight |
| Sensor | cheap fixed webcam from the 2000s | flagship Sony and Samsung smartphone camera sensors |
| Resolution | 1080p | 48MP-50MP images, higher than 8K UHD, which is around 33MP, around 23x increase in pixels |
| Model | SVM, sufficient for the simple task | HGBDT based model, finetuned parameters, training data weights, changed distribution of train/test data. Required for the more complex task|
| Test accuracy | 96% accuracy, very high, but much more simple task | 92.9% accuracy for the much more complex task |
| Feature extraction | more than 20 minutes | 42.6 sec with multithreaded feature extraction |
| Training time | 1 hour to 1.5 hours | 5 min 24 sec |
| CPU | i5 5200U notebook CPU, 2 core 4 threads | desktop R7 5700X CPU, 8 core 16 threads |
| CPU measured | baseline | single core 165% better performance, multi core 585% better, 40% lower memory latency |

---

## 2. Requirements

The project was made with Python `3.10.11`.

Package versions are pinned because there chould be compability issues, also the measured results were made with these versions.

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

The requirements notebook also has a version check cell to print the installed versions of the main packages.

---

## 3. Train

`train_apex_HGBDT.ipynb` prepares the training data, extracts features, trains the model and saves it.

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

Dimness means sunrise or dusk.

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

- the first HGBDT extraction took more than 30 minutes and was stopped because it took too much time
- with multithreading it takes 42.6 sec

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
- 5 fold stratified cross validation, trying to keep the original category ratio for the split
- `f1_macro` scoring, because macro F1 treats each class equally and is more useful for imbalanced data
- early stopping, so training stops when validation result stops improving
- saved final model with `joblib`

Main model settings:

| Setting | Value | Reason |
|---|---:|---|
| `max_iter` | 300 | maximum number of iterations; total trees are `max_iter * classes` |
| `early_stopping` | True | stop when there is no improvement |
| `validation_fraction` | 0.1 | 10% of training data used for early stopping validation |
| `n_iter_no_change` | 20 | stop if no improvement for 20 iterations |
| `class_weight` | custom dict | compensates imbalanced train category distribution |

Final hyperparameters:

| Parameter | Value | Reason |
|---|---:|---|
| `min_samples_leaf` | 30 | avoids decisions based on too few samples, which are often noise |
| `max_depth` | 16 | allows complex relations, but deeper trees can overfit |
| `learning_rate` | 0.06 | each tree contributes a smaller amount; early stopping helps control training |
| `l2_regularization` | 1 | penalty on leaf values to discourage overfitting |

The training notebook prints:

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
> still a remnant part of the SVM code, has to use the same range as in training

Overlay colors:

| Element | BGR color |
|---|---|
| sky | `(0, 150, 0)` |
| cloud | `(0, 0, 150)` |
| other | `(150, 0, 0)` |

There are 9 predicted categories, but only 3 main coloring classes: sky, cloud and other.

The final image is made by blending the original image and the colored overlay with 50-50 transparency.

The notebook prints:

- image size
- number of patches
- feature extraction time
- prediction time
- total time

---

## 5. Results

### 5.1 Original image/ APEX 1.0 / APEX 2.0

<p align="center">
  <img src="docs/assets/result_1_original_apex1_apex2.png" width="720" alt="Original vs APEX 1.0 vs APEX 2.0">
</p>

This comparison reflects on the original image, the thesis/APEX 1.0 result and the APEX 2.0 result. The first model is limited by fixed window views, weak webcam data, low resolution and the original SVM.Most `other` part of the images is classified as `sky`, while the `sky` is classified as `cloud` and so does the `clouds`. APEX 2.0 uses more diverse data, higher quality and resolution images and HGBDT classifier. That resulted a corretly identified `sky` part, only the `clouds` classified as `clouds`.The `other` part is mostly correctly evaluated, but at some parts, like the edge of the cloud is missclasified as that part dont have much training data (having a half mask, half cloud edge patch is ignored and discarded from the training data creation).

---

### 5.2 Original image / APEX 2.0

<p align="center">
  <img src="docs/assets/result_2_apex2_vs_original_model.png" width="720" alt="APEX 2.0 vs Original model result">
</p>

This comparison is demostrating that the more complex model has the ability to differentiate classes unreladet to position, unlike the first version. The `sky` part is corretly identified, and so does the `other` and `clouds`. The intresting part is that the clouds where mostly below the height of the mountain, but it don't mattered to the model, correctly recognised it above, the same and below mountain level.

---

### 5.3 APEX 2.0 vs Original — data, hardware and speed

<p align="center">
  <img src="docs/assets/result_3_apex2_vs_original_data_speed.png" width="720" alt="APEX 2.0 vs Original data, hardware and speed">
</p>

This comparison shows a very difficult classification problem. Image took at 10km height, has lower positioned high density clouds and higher low density ones. The model correctly recognise the wing of the plane as `other` and the higher density `clouds` most part (wrongly assumes the dark spot inside it as `other`). Also APEX 2.0 succesfully recognises, that the sun is not a bright `cloud` or `sky` part, but something in the `other` category. The sun surrounding is correctly recognised as `clouds` and so does below these the `sky` part. At higher parts the model struggles and missclassify the `clouds` to `sky` and `other` class.<img width="2736" height="1520" alt="apex 2 0" src="https://github.com/user-attachments/assets/2f838377-a815-4e2d-a827-343cbff90fd0" />
<img width="2736" height="1520" alt="apex 2 0" src="https://github.com/user-attachments/assets/5863ad21-0a2c-4899-8732-7ee7f791a663" />
