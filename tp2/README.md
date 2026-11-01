# TP2 — Computer Vision with PyTorch: CNN, Faster R-CNN and Vision Transformer

**Université Abdelmalek Essaâdi — Faculté des Sciences et Techniques de Tanger**  
**Département Génie Informatique | Cycle LSI | Deep Learning**  
**Professor:** Pr. ELAACHAK LOTFI  
**Student:** Saber Ben Hamda

## Objective

The objective of this lab is to learn how to build and train computer vision architectures with PyTorch in GPU mode. MNIST handwritten-digit recognition is used to study a custom CNN, Faster R-CNN, transfer learning with VGG16 and AlexNet, and a Vision Transformer implemented from scratch.

The experiments compare predictive performance, training behavior and computational cost. All values below are taken from the saved outputs of the repository notebooks; training was not rerun to write this report.

## Notebooks

| Part | Content | Notebook |
| --- | --- | --- |
| 1 | Custom CNN, Faster R-CNN, VGG16 and AlexNet | [Part 1 notebook](part1/Part1_MNIST_CNN_FasterRCNN_Transfer.ipynb) |
| 2 | Vision Transformer from scratch | [Part 2 notebook](part2/Part2_MNIST_ViT_From_Scratch.ipynb) |

[Open Part 1 in Colab](https://colab.research.google.com/github/saberBenhamda0/Deep_learning_-_Applications_assignments/blob/main/tp2/part1/Part1_MNIST_CNN_FasterRCNN_Transfer.ipynb) · [Open Part 2 in Colab](https://colab.research.google.com/github/saberBenhamda0/Deep_learning_-_Applications_assignments/blob/main/tp2/part2/Part2_MNIST_ViT_From_Scratch.ipynb)

## 1. Dataset and experimental protocol

MNIST contains grayscale images of handwritten digits from **0 to 9**, with a resolution of **28 × 28 pixels**. The notebooks download MNIST using `torchvision.datasets.MNIST`, without requiring Kaggle credentials.

The original 60,000 training images are split using stratified sampling and seed **42**. The official 10,000-image test set is kept separate from training and checkpoint selection.

### Configuration of the saved runs

| Setting | Part 1 | Part 2: ViT |
| --- | --- | --- |
| Quick mode | `False` | `False` |
| Training images | 54,000 | 54,000 |
| Validation images | 6,000 | 6,000 |
| Test images | 10,000 | 10,000 |
| Epochs | 10 | 10 |
| GPU | Tesla T4 | Tesla T4 |
| PyTorch | 2.11.0+cu130 | 2.11.0+cu130 |
| Torchvision | 0.26.0+cu130 | 0.26.0+cu130 |

Both parts use the same seeded training/validation split, official test set, epoch count and GPU type. The updated Part 2 notebook successfully combines all five models in its results table. This improves comparability, although input resolutions, pretraining, architectures and computational costs still differ.

For classifiers, the checkpoint with the lowest validation cross-entropy is restored before test evaluation. Faster R-CNN uses the checkpoint with the highest validation classification accuracy.

## 2. Part 1 — CNN classifier and transfer learning

### Step 1: Prepare the data

Images are converted into tensors. The custom CNN uses single-channel images normalized with mean **0.1307** and standard deviation **0.3081**. Data loaders provide training, validation and test batches, with shuffling enabled only for training.

### Step 2: Build and train a custom CNN

The CNN is trained from scratch with the following architecture:

| Layer | Configuration | Output per image |
| --- | --- | --- |
| Input | Grayscale digit | 1 × 28 × 28 |
| Convolution + ReLU | 32 filters, kernel 3 × 3, stride 1, padding 1 | 32 × 28 × 28 |
| Max pooling | Kernel 2 × 2, stride 2 | 32 × 14 × 14 |
| Convolution + ReLU | 64 filters, kernel 3 × 3, stride 1, padding 1 | 64 × 14 × 14 |
| Max pooling | Kernel 2 × 2, stride 2 | 64 × 7 × 7 |
| Flatten | Convert feature maps into a vector | 3,136 |
| Fully connected + ReLU | 128 units | 128 |
| Dropout | Probability 0.3 | 128 |
| Output layer | 10 logits | 10 |

Training uses cross-entropy, AdamW with learning rate **0.001**, weight decay **0.0001**, and batch size **128**.

**Observed behavior:** training loss falls from **0.2300** to **0.0144**. The lowest validation loss is approximately **0.0400** at epoch **9**, which is selected for testing. Training loss continues decreasing at epoch 10 while validation loss rises, suggesting mild late overfitting.

### Step 3: Adapt MNIST to Faster R-CNN

Faster R-CNN performs object detection, so each MNIST image is represented as one digit with a bounding box. The box is calculated from the minimum and maximum coordinates of nonzero pixels. These boxes are derived automatically from the images rather than manually annotated.

The grayscale image is repeated across three channels and resized internally to **64 × 64**. Labels are shifted from **0–9** to **1–10**, because Faster R-CNN reserves label **0** for background.

The detector uses a randomly initialized backbone with two convolution layers, a Region Proposal Network, ROI Align and Faster R-CNN prediction heads. Anchor sizes are **8, 16 and 32**, with aspect ratios **0.5, 1 and 2**. Training uses AdamW, learning rate **0.001**, weight decay **0.0001**, and batch size **32**.

For classification evaluation, the highest-scoring detection provides the image label. A missing detection is assigned **−1** and counted as an error. Localization is evaluated using mean Intersection over Union (IoU) and the proportion of images with both a correct class and IoU ≥ 0.5.

**Observed behavior:** the composite training loss decreases from **0.4601** to **0.1955**, while validation accuracy increases from **96.55%** to **99.13%**. The best validation checkpoint is epoch **10**.

### Step 4: Fine-tune VGG16 and AlexNet

Both models start from **ImageNet-pretrained weights**. MNIST images are resized to **224 × 224**, converted into three identical channels, and normalized using ImageNet statistics.

The final classification layer is replaced with a **10-output layer**. Earlier parameters are frozen; the last convolution block and the new output layer are trained. This is fine-tuning because convolutional weights are updated, rather than using the network only as a fixed feature extractor.

Both models use AdamW with learning rate **0.0001**, weight decay **0.0001**, and batch size **32**.

**Observed behavior:** VGG16 reaches its lowest validation loss, **0.0214**, at epoch **7**. AlexNet reaches its lowest validation loss, **0.0237**, at epoch **10**. VGG16 shows a growing training/validation loss gap in later epochs, supporting the use of validation-based checkpoint selection.

### Step 5: Compare the Part 1 results

Accuracy and macro-F1 are shown as percentages. Training time includes training and validation, but excludes pretrained-weight downloads and final test evaluation.

| Model | Test accuracy | Macro-F1 | Test cross-entropy | Training time (s) | Approx. time (min) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Custom CNN | 99.15% | 99.1413% | 0.028509 | 132.58 | 2.21 |
| VGG16 | **99.34%** | **99.3396%** | 0.021270 | 3,540.99 | 59.02 |
| AlexNet | 99.31% | 99.3035% | **0.020648** | 1,014.24 | 16.90 |
| Faster R-CNN | 99.27% | 99.2622% | Not applicable | 4,001.54 | 66.69 |

Faster R-CNN additionally obtains:

| Localization metric | Result |
| --- | ---: |
| Mean IoU of the selected detection | 0.943441 |
| Correct class and IoU ≥ 0.5 | 99.24% |

These localization metrics are **not COCO mAP**. The detector's training objective combines classification, box regression, objectness and proposal regression losses. Its numerical value cannot be ranked directly against classifier test cross-entropy.

### Part 1 interpretation and conclusion

- **VGG16 achieves the highest measured accuracy**, correctly classifying 9,934 test images. Its gain over the CNN is **0.19 percentage points**, or 19 additional correct predictions, for approximately **26.7 times** the measured training time.
- **AlexNet is close to VGG16**, with only three fewer correct test predictions and approximately **3.5 times shorter** training time. It also has the lowest classifier test cross-entropy, illustrating that accuracy and loss do not necessarily rank models identically.
- **The custom CNN provides the best practical speed–accuracy tradeoff in this run.** It reaches 99.15% accuracy in approximately 2.2 minutes, using the native image resolution and no external pretraining.
- **Faster R-CNN successfully localizes and recognizes the digits**, but takes approximately **30.2 times** the CNN training time. For single, already-centered digits, its localization machinery adds substantial cost without giving the highest classification accuracy.

The results favor the small CNN when efficiency matters, and VGG16 when choosing solely by the highest observed accuracy. Differences this small should be confirmed over multiple seeds before claiming a reliable architectural advantage.

## 3. Part 2 — Vision Transformer from scratch

### Step 1: Define the ViT architecture

The ViT is implemented directly in PyTorch without a pretrained transformer or `nn.TransformerEncoder`.

| Component | Configuration |
| --- | --- |
| Input | 1 × 28 × 28 |
| Patch size | 4 × 4 |
| Number of image patches | 49 |
| Embedding dimension | 64 |
| Class token | One learnable token |
| Positional embeddings | Learnable embeddings for 50 tokens |
| Encoder blocks | 4 |
| Attention heads per block | 4 |
| MLP hidden dimension | 128 |
| Activation | GELU |
| Normalization | Pre-norm LayerNorm and final LayerNorm |
| Dropout | 0.1 |
| Classification output | 10 logits from the class token |
| Total parameters | 139,018 |

Each image is split into patches and projected into embeddings. After adding the class token and positional embeddings, transformer blocks apply multi-head self-attention and feed-forward layers with residual connections. The final class token is used to predict the digit.

### Step 2: Train and evaluate

The full run uses AdamW with learning rate **0.001**, weight decay **0.0001**, batch size **128**, and **10 epochs** over **54,000 training images**.

| Epoch | Training loss | Training accuracy | Validation loss | Validation accuracy |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1.1305 | 59.41% | 0.4735 | 84.63% |
| 2 | 0.4029 | 87.07% | 0.2370 | 92.52% |
| 3 | 0.2641 | 91.66% | 0.1741 | 94.53% |
| 4 | 0.2044 | 93.56% | 0.1406 | 95.78% |
| 5 | 0.1665 | 94.68% | 0.1353 | 96.07% |
| 6 | 0.1430 | 95.47% | 0.1183 | 96.75% |
| 7 | 0.1309 | 95.80% | 0.1132 | 96.73% |
| 8 | 0.1141 | 96.37% | 0.0986 | 97.38% |
| 9 | 0.1073 | 96.56% | 0.0931 | 97.27% |
| 10 | 0.0972 | 96.89% | 0.0920 | 96.85% |

| Test metric | ViT result |
| --- | ---: |
| Accuracy | 97.50% |
| Macro-F1 | 97.4710% |
| Cross-entropy | 0.083976 |
| Training time | 205.40 seconds (3.42 minutes) |

Training loss decreases from **1.1305** to **0.0972**, while validation loss decreases from **0.4735** to **0.0920**. The lowest validation loss occurs at epoch **10**, so this checkpoint is restored for testing. Validation accuracy peaks at **97.38%** at epoch **8**, but checkpoint selection is based on loss rather than accuracy. These metrics need not improve together because cross-entropy also reflects prediction confidence.

The final test accuracy corresponds to **9,750 correct predictions** and **250 errors** out of 10,000 images. Macro-F1 is close to accuracy, indicating strong aggregate performance across digit classes, although it does not imply identical performance for every class. The notebook provides a confusion matrix, per-class report and examples of misclassified digits for closer inspection.

Validation loss continues improving near the end of training, so additional training or hyperparameter tuning may be useful. This trend alone does not guarantee further improvements on unseen data.

### Step 3: Compare with Part 1

All five saved runs now use 54,000 training images, 6,000 validation images, 10,000 test images and 10 epochs on a Tesla T4.

| Model | Test accuracy | Macro-F1 | Test cross-entropy | Training time (s) |
| --- | ---: | ---: | ---: | ---: |
| Custom CNN | 99.15% | 99.1413% | 0.028509 | **132.58** |
| VGG16 | **99.34%** | **99.3396%** | 0.021270 | 3,540.99 |
| AlexNet | 99.31% | 99.3035% | **0.020648** | 1,014.24 |
| Faster R-CNN | 99.27% | 99.2622% | Not applicable | 4,001.54 |
| ViT | 97.50% | 97.4710% | 0.083976 | 205.40 |

**The custom CNN outperforms the ViT in both accuracy and measured training time in this experiment.** Its accuracy is **1.65 percentage points higher**, corresponding to **165 fewer test errors**. ViT takes approximately **1.55 times** the CNN training time, while remaining faster than AlexNet, VGG16 and Faster R-CNN.

VGG16 remains the most accurate model, exceeding ViT by **1.84 percentage points**, but takes approximately **17.24 times** its training time. The results therefore show a clear tradeoff: the lightweight CNN is the most efficient high-accuracy solution in these runs, while VGG16 gives the highest observed accuracy at much greater cost.

CNNs build in locality and spatial weight sharing, while the ViT learns relationships between image patches through self-attention. The observed results are consistent with these CNN assumptions being useful for this small-image task. However, a single configuration and seed do not establish that CNNs always outperform transformers. The ViT is trained from scratch, whereas VGG16 and AlexNet benefit from ImageNet pretraining; matching image counts and epochs does not equalize pretraining or compute.

### Part 2 conclusion

The ViT implementation successfully learns handwritten-digit recognition and achieves **97.50% test accuracy**. For the tested architectures and settings, it does not surpass the CNN-based models. Further work could investigate learning-rate schedules, additional epochs, patch sizes and model dimensions, using validation data for selection before a final test evaluation.

## 4. Reproduce the experiments

1. Open the Part 1 notebook in Google Colab and select **Runtime → Change runtime type → GPU**.
2. Set `QUICK=False` to reproduce the full Part 1 configuration, then run all cells in order.
3. Download the generated output ZIP before the runtime disconnects. It includes checkpoints, training histories, plots, results and configuration.
4. Open Part 2 and set `QUICK=False` to reproduce the saved full run with the same data sizes and epoch count as Part 1.
5. Run Part 2 and upload the Part 1 CSV and configuration when requested. The comparison cell checks compatibility before merging results.
6. Save the executed notebooks and relevant result files in the repository. Large datasets and model checkpoints do not need to be committed.

The saved outputs contain DataLoader worker-cleanup warnings (`can only test a child process`), but final metrics were produced. If these warnings recur, setting `num_workers=0` in the data loaders is a practical workaround. Changing loader settings can affect runtime measurements.

## 5. Limitations

The experiments use one random seed and one saved run per model. Transfer models benefit from ImageNet pretraining and use larger inputs, so their comparison with the CNN is not an equal-compute or equal-pretraining study. Faster R-CNN solves an additional localization task using image-derived boxes. Validation computations also differ between detectors and classifiers.

Both parts now match data sizes and epoch counts, but the models still differ in optimization settings, input sizes, pretraining and computational cost. No claim of statistical significance or universal CNN-versus-transformer superiority is made from these single-seed results.

## 6. Brief learning synthesis

Through this lab, I learned how to prepare image data, define neural architectures in PyTorch, train them on a GPU and evaluate their generalization using a separate test set. I practiced building a CNN, adapting a classification dataset for Faster R-CNN, fine-tuning pretrained networks and implementing patch embeddings and multi-head attention for a ViT.

The results taught me to consider computational cost alongside accuracy: a small CNN can perform very well on a simple dataset, while a larger model may offer only a small improvement at a much higher cost. I also learned that meaningful comparisons require matching experimental settings and that different loss functions should not be compared as if they measured the same objective. The full ViT run achieved 97.50% accuracy, while the small CNN achieved 99.15% with less measured training time. My next step would be to tune the ViT using validation data and repeat the experiments with several seeds.

## References and result provenance

- [MNIST dataset specified in the assignment](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
- [Vision Transformer tutorial specified in the assignment](https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c)
- [Original Vision Transformer paper](https://arxiv.org/abs/2010.11929)
- [Torchvision documentation](https://docs.pytorch.org/vision/stable/index.html)
- Results and implementation reviewed from repository commit [`f29555b`](https://github.com/saberBenhamda0/Deep_learning_-_Applications_assignments/tree/f29555b8e710b33e9bfa5ba706007ec3a561fae7/tp2).
