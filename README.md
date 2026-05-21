<h2>TensorFlow-FlexUNet-Image-Segmentation-OASBUD-Breast-Ultrasound (2026/05/21)</h2>
Sarah T. Arai<br>
Software Laboratory antillia.com<br><br>
This is the first experiment of Image Segmentation for <b>OASBUD-Breast-US (Benign and Malignant)</b> based on 
our <a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet</a>
 (<b>TensorFlow Flexible UNet Image Segmentation Model for Multiclass</b>), and a 682x512 pixels PNG
 <a href="https://drive.google.com/file/d/1By9LkkP1RJBLMVPK7zzhuilFGHcg3_g2/view?usp=sharing">
Augmented-OASBUD-ImageMask-Dataset.zip</a>, which was derived by us from <br><br>
<a href="https://www.kaggle.com/datasets/soumikmondol/oasbud-org">
<b>oasbud_org</b>
</a> by Soumik_Shanto_Mondol.
<br><br>
<hr>
<b>Actual Image Segmentation for OASBUD Images of 682x512 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained by the dataset appear similar to 
the ground truth masks.
<br><br>
<b>class-color-map = {Benign:green, Malignant:red}</b>
<br><br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/OASBUD/mini_test/images/barrdistorted_1003_0.3_0.3_Benign_10093.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/OASBUD/mini_test/masks/barrdistorted_1003_0.3_0.3_Benign_10093.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/OASBUD/mini_test_output/barrdistorted_1003_0.3_0.3_Benign_10093.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/OASBUD/mini_test/images/barrdistorted_1001_0.3_0.3_Malignant_10013.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/OASBUD/mini_test/masks/barrdistorted_1001_0.3_0.3_Malignant_10013.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/OASBUD/mini_test_output/barrdistorted_1001_0.3_0.3_Malignant_10013.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/OASBUD/mini_test/images/barrdistorted_1003_0.3_0.3_Malignant_10084.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/OASBUD/mini_test/masks/barrdistorted_1003_0.3_0.3_Malignant_10084.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/OASBUD/mini_test_output/barrdistorted_1003_0.3_0.3_Malignant_10084.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>1. Dataset Citation</h3>
The dataset used here was taken from <br><br>
<a href="https://www.kaggle.com/datasets/soumikmondol/oasbud-org">
<b>oasbud_org</b>
</a> by Soumik_Shanto_Mondol.
<br>
<br>
For more information of the original <b>OASBUD</b>
, please refer to
<a href="https://zenodo.org/records/545928">
Open access database of raw ultrasonic signals acquired from malignant and benign breast lesions</a>
<br><br>
The dataset presented in: H. Piotrzkowska-Wróblewska, K. Dobruch-Sobczak, M. Byra, A. Nowicki,<br>
 "Open access database of raw ultrasonic signals acquired from malignant and benign breast lesions",<br>
 Medical Physics, http://dx.doi.org/10.1002/mp.12538. <br>
<br>
<b>License</b><br>
Unknown
<br>
<br>
<h3>
2 OASBUD ImageMask Dataset
</h3>
<h3>2.1 OASBUD ImageMask Dataset</h3>
 If you would like to train this OASBUD Segmentation model by yourself,
 please download the dataset from the google drive  
 <a href="https://drive.google.com/file/d/1By9LkkP1RJBLMVPK7zzhuilFGHcg3_g2/view?usp=sharing">
Augmented-OASBUD-ImageMask-Dataset.zip</a> 
, expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be
<br>
<pre>
./dataset
└─OASBUD
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
<b>OASBUD Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/OASBUD/OASBUD_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is large enough to use for the
 training set of our segmentation model.
<br>
<br>
<h3>2.2 Derivation of OASBUD ImageMask Dataset</h3>
The folder structure of the original <b>oasbud_org</b> dataset is the following.<br>
<pre>
./OASBUD
├─BMODE
│  ├─Benign
│  │  ├─Fold1
│  │  ├─Fold2
│  │  ├─Fold3
│  │  ├─Fold4
│  │  └─Fold5
│  └─Malignant
│      ├─Fold1
│      ├─Fold2
│      ├─Fold3
│      ├─Fold4
│      └─Fold5
├─HSCAN
...
├─MASK
│  ├─Benign
│  │  ├─Fold1
│  │  ├─Fold2
│  │  ├─Fold3
│  │  ├─Fold4
│  │  └─Fold5
│  └─Malignant
│      ├─Fold1
│      ├─Fold2
│      ├─Fold3
│      ├─Fold4
│      └─Fold5
└─NAKAGAMI
...
</pre>
We generated a 682x512 pixels resized master PNG ImageMask dataset from the BMP images in <b>BMODE/Benign</b> and 
<b>BMODE/Malignant</b> folders and their corresponding BMP masks in <b>MASK/Benign</b> and <b>Mask/Malignant</b> folders,
and then our augmented dataset from the master by using the following image deformation tools.<br>
<a href="https://github.com/sarah-antillia/Image-Deformation-Tool">Image-Deformation-Tool</a><br>
<a href="https://github.com/sarah-antillia/Image-Distortion-Tool">Image-Distortion-Tool</a> <br>
<a href="https://github.com/sarah-antillia/Barrel-Image-Distortion-Tool">Barrel-Image-Distortion-Tool</a> <br>
<br>
<h3>2.3 Train Sample Images and Masks</h3>

<b>Train sample images</b><br>
<img src="./projects/TensorFlowFlexUNet/OASBUD/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train sample masks</b><br>
<img src="./projects/TensorFlowFlexUNet/OASBUD/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<h3>
3 Train TensorFlowFlexUNet Model
</h3>
 We trained OASBUD TensorFlowFlexUNet Model by using the 
<a href="./projects/TensorFlowFlexUNet/OASBUD/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/OASBUD and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters = 16 </b> and large <b>base_kernels = (11,11)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large <b>num_layers = 8</b> (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
;You may specify your own UNet class derived from our TensorFlowFlexModel
model         = "TensorFlowFlexUNet"
image_width    = 512
image_height   = 512
image_channels = 3
input_normalize = True
normalization  = False
num_classes    = 3
base_filters   = 16
base_kernels   = (11,11)
num_layers     = 8
dropout_rate   = 0.05
dilation       = (1,1)
</pre>
<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>
<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b>Dataset class</b><br>
Specifed <a href="./src/ImageCategorizedMaskDataset.py">ImageCategorizedMaskDataset</a> class.<br>
<pre>
[dataset]
class_name    = "ImageCategorizedMaskDataset"
</pre>
<br>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>
<b>RGB Color map</b><br>
Specifed rgb color map dict for OASBUD 1+2 classes.<br>
<pre>
[mask]
mask_datatyoe    = "categorized"
mask_file_format = ".png"
;OASBUDrgb color map dict for 1+2 classes.
;                      Benign:green,  Malignant:red
rgb_map = {(0,0,0):0, (0, 255, 0):1, (255,0,0):2, }
</pre>
<b>Epoch change inference callback</b><br>
Enabled <a href="./src/EpochChangeInferencer.py">epoch_change_infer callback</a></b>.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
num_infer_images         = 6
</pre>
By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> 
<br> 
As shown below, early in the model training, the predicted masks from our UNet segmentation model showed 
discouraging results.
 However, as training progressed through the epochs, the predictions gradually improved. 
 <br> 
<br>
<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/OASBUD/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 18,19,20)</b><br>
<img src="./projects/TensorFlowFlexUNet/OASBUD/asset/epoch_change_infer_at_middle.png" width="1024" height="auto"><br>
<br>

<b>Epoch_change_inference output at ending (epoch 38,39,40)</b><br>
<img src="./projects/TensorFlowFlexUNet/OASBUD/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>

In this experiment, the training process was terminated at epoch 40.<br><br>
<img src="./projects/TensorFlowFlexUNet/OASBUD/asset/train_console_output_at_epoch40.png" width="1024" height="auto"><br>
<br>

<a href="./projects/TensorFlowFlexUNet/OASBUD/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/OASBUD/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/OASBUD/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/OASBUD/eval/train_losses.png" width="520" height="auto"><br>
<br>
<h3>
4 Evaluation
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/OASBUD</b> folder,
and run the following bat file to evaluate TensorFlowUNet model for OASBUD.<br>
<pre>
>./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/OASBUD/asset/evaluate_console_output_at_epoch40.png" width="1024" height="auto">
<br><br>Image-Segmentation-OASBUD

<a href="./projects/TensorFlowFlexUNet/OASBUD/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_crossentropy) to this <b>OASBUD/test</b> was very low and dice_coef_multiclass very high as shown below.
<br>
<pre>
categorical_crossentropy,0.0083
dice_coef_multiclass,0.9955
</pre>
<br>
<h3>
5 Inference
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/OASBUD</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorFlowUNet model for OASBUD.<br>
<pre>
>./3.infer.bat
</pre>
This simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/OASBUD/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/OASBUD/asset/mini_test_masks.png" width="1024" height="auto"><br>
<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/OASBUD/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks of OASBUD Images of 682x512 pixels </b><br>
As shown below, the inferred masks predicted by our segmentation model trained by the dataset appear similar 
to the ground truth masks.
<br><br>
<b>class-color-map = {Benign:green, Malignant:red}</b>
<br><br>
<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/OASBUD/mini_test/images/Benign_10026.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/OASBUD/mini_test/masks/Benign_10026.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/OASBUD/mini_test_output/Benign_10026.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/OASBUD/mini_test/images/Benign_10082.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/OASBUD/mini_test/masks/Benign_10082.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/OASBUD/mini_test_output/Benign_10082.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/OASBUD/mini_test/images/deformed_alpha_1300_sigmoid_10_Benign_10045.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/OASBUD/mini_test/masks/deformed_alpha_1300_sigmoid_10_Benign_10045.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/OASBUD/mini_test_output/deformed_alpha_1300_sigmoid_10_Benign_10045.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/OASBUD/mini_test/images/Malignant_10001.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/OASBUD/mini_test/masks/Malignant_10001.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/OASBUD/mini_test_output/Malignant_10001.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/OASBUD/mini_test/images/Malignant_10023.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/OASBUD/mini_test/masks/Malignant_10023.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/OASBUD/mini_test_output/Malignant_10023.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/OASBUD/mini_test/images/deformed_alpha_1300_sigmoid_8_Malignant_10103.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/OASBUD/mini_test/masks/deformed_alpha_1300_sigmoid_8_Malignant_10103.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/OASBUD/mini_test_output/deformed_alpha_1300_sigmoid_8_Malignant_10103.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
References
</h3>
<b>1. TensorFlow-FlexUNet-Image-Segmentation-Breast-Ultrasound-Images</b><br>
Toshiyuki Arai<br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Breast-Ultrasound-Images">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Breast-Ultrasound-Images</a>
<br><br>

<b>2. TensorFlow-FlexUNet-Image-Segmentation-Breast-Lesions-USG</b><br>
Toshiyuki Arai<br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Breast-Lesions-USG">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Breast-Lesions-USG</a>
<br><br>
<b>3. TensorFlow-FlexUNet-Image-Segmentation-Model</b><br>
Toshiyuki Arai<br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model</a>
<br><br>

