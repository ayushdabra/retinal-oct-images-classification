# Detecting Retina Damage from Optical Coherence Tomography (OCT) Images
## Context
Retinal Optical Coherence Tomography (OCT) is an imaging technique used to capture high-resolution cross sections of the retinas of living patients. Approximately 30 million OCT scans are performed each year, and the analysis and interpretation of these images takes up a significant amount of time (Swanson and Fujimoto, 2017).
<p align="center">
  <img src="/assets/dataset_classes.png">
</p>

<p align="center">Figure 1. Representative Optical Coherence Tomography Images and the Workflow Diagram [Kermany et. al. 2018]</p>

(A) (Far left) Choroidal Neo-Vascularization (CNV) with neovascular membrane (white arrowheads) and associated subretinal fluid (arrows). (Middle left) Diabetic Macular Edema (DME) with retinal-thickening-associated intraretinal fluid (arrows). (Middle right) Multiple drusen (arrowheads) present in early AMD. (Far right) Normal retina with preserved foveal contour and absence of any retinal fluid/edema.

## Getting Started
The <a href= "/vgg16-for-retinal-oct-images-dataset.ipynb">`vgg16-for-retinal-oct-images-dataset.ipynb`</a> notebook can be directly run on Kaggle after loading the dataset in the Kaggle Kernel. Use Kaggle's Nvidia Tesla P100 GPU for faster training and evaluation.
### Pre-Requisites
For running the notebook on your local machine, following pre-requisites must be satisfied:
- Python3
- NumPy
- Pandas
- Scikit-learn
- Scikit-image
- Seaborn
- IPython
- Matplotlib
- Tensorflow 2.0
- Keras
- Keract

### Installation
**Dependencies:**
```
# With Tensorflow CPU
pip install -r requirements.txt

# With Tensorflow GPU
pip install -r requirements-gpu.txt
```
**Nvidia Driver (For GPU, if you haven't set it up already):**
```
# Ubuntu 20.04
sudo apt-add-repository -r ppa:graphics-drivers/ppa
sudo apt install nvidia-driver-430

# Windows/Other
https://www.nvidia.com/Download/index.aspx
```
## Dataset
* The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (NORMAL, CNV, DME, DRUSEN). There are 84,495 X-Ray images (JPEG) and 4 categories (NORMAL, CNV, DME, DRUSEN).
* Images are labeled as (disease)-(randomized patient ID)-(image number by this patient) and split into 4 directories: CNV, DME, DRUSEN, and NORMAL.
* Optical coherence tomography (OCT) images (Spectralis OCT, Heidelberg Engineering, Germany) were selected from retrospective cohorts of adult patients from the Shiley Eye Institute of the University of California San Diego, the California Retinal Research Foundation, Medical Center Ophthalmology Associates, the Shanghai First People’s Hospital, and Beijing Tongren Eye Center between July 1, 2013 and March 1, 2017.
### Sample Images
|![](/assets/normal_eye.png) NORMAL|![](/assets/cnv_eye.png) CNV|
:-------------------------:|:-------------------------:
![](/assets/dme_eye.png) **DME**|![](/assets/drusen_eye.png) **DRUSEN**

### Image Histogram (Tonal Distribution)
Histogram of a normal retina image in the train dataset:
<p align="center">
  <img src= "/assets/histogram.jpeg">
</p>

## Approach
### Image Data Augmentation
Data augmentation is done through the following techniques:
- Rescaling (1./255)
- Zoom (0.73-0.9)
- Horizontal Flipping
- Rotation (10)
- Width Shifting (0.10)
- Fill Mode (constant)
- Height Shifting (0.10)
- Brightness (0.55-0.9)

<p><img src= "/assets/augmented_image.jpeg"></p>

### Model Details
Tranfer learning has been used on VGG16 CNN Architecture pre-trained on ImageNet dataset, with a custom classifier having a dropout(0.2) layer, and a fully-connected dense(4) layer with softmax activation.

A detailed layout of the model is available <a href= "/assets/model_plot.png">here.</a>

### Training Results

|Model|Epochs|Train Accuracy|Train Loss| Val Accuracy| Val Loss |
| :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
| Baseline | 10 | 85.16 % | 0.4144 | 84.37 % | 0.3519 |
| Finetuned | 10 | 97.01 % | 0.086 | 100 % | 0.0309|

| ![](/assets/baseline_acc_epoch.jpeg) | ![](/assets/baseline_loss_epoch.jpeg) |
| :-----: | :-----: |
| ![](/assets/finetuned_acc_epoch.jpeg) | ![](/assets/finetuned_loss_epoch.jpeg) |

The `baseline_training_csv.log` and `finetuned_training_csv.log` files contain epoch wise training details of the baseline model and the finetuned model respectively.

### Evaluation on Test Dataset
The model performed really well and achieved an accuracy of 98.96 % on the test set.

**Confusion Matrix:**

![](/assets/confusion_matrix.jpeg)

**Classification Report:**
```
              precision    recall  f1-score   support

         CNV       0.96      1.00      0.98       242
         DME       1.00      1.00      1.00       242
      DRUSEN       1.00      0.96      0.98       242
      NORMAL       1.00      1.00      1.00       242

    accuracy                           0.99       968
   macro avg       0.99      0.99      0.99       968
weighted avg       0.99      0.99      0.99       968
```

## Model Layers' Activations Visualization
|![](/assets/1_block1_conv1.jpeg) block1_conv1|![](/assets/4_block2_conv1.jpeg) block2_conv1|
:-------------------------:|:-------------------------:
![](/assets/7_block3_conv1.jpeg) **block3_conv1**|![](/assets/11_block4_conv1.jpeg) **block4_conv1**
![](/assets/15_block5_conv1.jpeg) **block5_conv1**|![](/assets/16_block5_conv2.jpeg) **block5_conv2**

* Activation Maps of model's layers are available in the <a href= "/activations">`activations`</a> directory.
* Heatmaps of model's layers are available in the <a href= "/heatmaps">`heatmaps`</a> directory.

## References
* Data- https://data.mendeley.com/datasets/rscbjbr9sj/2
* Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), “Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification”, Mendeley Data, V2, doi: 10.17632/rscbjbr9sj.2
* Daniel S. Kermany, Michael Goldbaum, Wenjia Cai, Carolina C.S. Valentim, Huiying Liang, Sally L. Baxter, Alex McKeown, Ge Yang, Xiaokang Wu, Fangbing Yan, Justin Dong, Made K. Prasadha, Jacqueline Pei, Magdalene Y.L. Ting, Jie Zhu, Christina Li, Sierra Hewett, Jason Dong, Ian Ziyar, Alexander Shi, Runze Zhang, Lianghong Zheng, Rui Hou, William Shi, Xin Fu, Yaou Duan, Viet A.N. Huu, Cindy Wen, Edward D. Zhang, Charlotte L. Zhang, Oulan Li, Xiaobo Wang, Michael A. Singer, Xiaodong Sun, Jie Xu, Ali Tafreshi, M. Anthony Lewis, Huimin Xia, Kang Zhang, "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning", Kermany et al., 2018, Cell; February 22, 2018 Copyright (c) 2018 Elsevier Inc. <a href= "https://www.cell.com/action/showPdf?pii=S0092-8674%2818%2930154-5">[PDF]</a>
* Karen Simonyan and Andrew Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition", arXiv:1409.1556, 2014. <a href= "https://arxiv.org/pdf/1409.1556v6.pdf">[PDF]</a>
* Keract by Philippe Rémy <a href= "https://github.com/philipperemy/keract">(@github/philipperemy)</a> used under the IT License Copyright (c) 2019.
