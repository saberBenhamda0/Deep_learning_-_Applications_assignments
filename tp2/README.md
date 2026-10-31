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
| Quick mode | `False` | `True` |
| Training images | 54,000 | 6,000 |
| Validation images | 6,000 | 1,000 |
| Test images | 10,000 | 10,000 |
| Epochs | 10 | 3 |
| GPU | Tesla T4 | Tesla T4 |
| PyTorch | 2.11.0+cu130 | 2.11.0+cu130 |
| Torchvision | 0.26.0+cu130 | 0.26.0+cu130 |

**The ViT result is a pilot experiment, not an equal-budget comparison with Part 1.** Its smaller training set is a stratified subset of the training partition. The Part 2 comparison cell correctly rejects combining the runs because `quick`, `epochs`, `train_n` and `val_n` differ.

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

The saved pilot uses AdamW with learning rate **0.001**, weight decay **0.0001**, and batch size **128**.

| Epoch | Training loss | Training accuracy | Validation loss | Validation accuracy |
| --- | ---: | ---: | ---: | ---: |
| 1 | 2.0092 | 26.28% | 1.6909 | 37.10% |
| 2 | 1.6352 | 39.55% | 1.4540 | 45.90% |
| 3 | 1.4195 | 48.72% | 1.1522 | 59.10% |

| Test metric | ViT pilot result |
| --- | ---: |
| Accuracy | 60.88% |
| Macro-F1 | 60.0109% |
| Cross-entropy | 1.141775 |
| Training time | 7.68 seconds |

Validation loss decreases throughout the three epochs and validation accuracy continues to improve. This suggests that the pilot has not converged and warrants longer training; it does not demonstrate that the ViT architecture is intrinsically unsuitable for MNIST.

The per-class report shows strong variation: digit **1** has an F1 score of approximately **93.30%**, while digits **5** and **2** obtain approximately **37.47%** and **41.60%**. The model therefore recognizes some classes much more reliably than others. The notebook includes a confusion matrix and misclassified examples for further inspection.

### Step 3: Compare with Part 1

The saved Part 1 models achieve over **99%** accuracy, while the saved ViT pilot reaches **60.88%**. However, Part 1 uses **nine times more training images** and **more than three times as many epochs**. The ViT's shorter runtime must therefore not be interpreted as evidence that it is the most efficient model at a comparable accuracy.

CNNs encode useful locality and weight-sharing assumptions, whereas the ViT learns relationships between image patches through attention. These architectural differences may affect sample efficiency, but this experiment does not isolate their effect because the training budgets differ.

To complete a controlled comparison, rerun Part 2 with `QUICK=False`, giving it 54,000 training images, 6,000 validation images and 10 epochs. Use the same GPU type and upload the matching Part 1 `part1_results.csv` and `config.json` files. Further tuning and repeated runs may still be needed.

The Part 2 output stating **“Best measured accuracy: ViT”** refers only to the one-row table remaining after the incompatible Part 1 results were rejected. It does **not** mean ViT outperformed the Part 1 models.

## 4. Reproduce the experiments

1. Open the Part 1 notebook in Google Colab and select **Runtime → Change runtime type → GPU**.
2. Set `QUICK=False` to reproduce the full Part 1 configuration, then run all cells in order.
3. Download the generated output ZIP before the runtime disconnects. It includes checkpoints, training histories, plots, results and configuration.
4. Open Part 2. Keep `QUICK=True` to reproduce the saved pilot, or set `QUICK=False` for a comparison using the same data sizes and epoch count as Part 1.
5. Run Part 2 and upload the Part 1 CSV and configuration when requested. The comparison cell checks compatibility before merging results.
6. Save the executed notebooks and relevant result files in the repository. Large datasets and model checkpoints do not need to be committed.

The saved outputs contain DataLoader worker-cleanup warnings (`can only test a child process`), but final metrics were produced. If these warnings recur, setting `num_workers=0` in the data loaders is a practical workaround. Changing loader settings can affect runtime measurements.

## 5. Limitations

The experiments use one random seed and one saved run per model. Transfer models benefit from ImageNet pretraining and use larger inputs, so their comparison with the CNN is not an equal-compute or equal-pretraining study. Faster R-CNN solves an additional localization task using image-derived boxes. Validation computations also differ between detectors and classifiers.

The ViT pilot has a different training budget and cannot support a definitive CNN-versus-transformer conclusion. No claim of statistical significance is made for the small accuracy differences among Part 1 models.

## 6. Brief learning synthesis

Through this lab, I learned how to prepare image data, define neural architectures in PyTorch, train them on a GPU and evaluate their generalization using a separate test set. I practiced building a CNN, adapting a classification dataset for Faster R-CNN, fine-tuning pretrained networks and implementing patch embeddings and multi-head attention for a ViT.

The results taught me to consider computational cost alongside accuracy: a small CNN can perform very well on a simple dataset, while a larger model may offer only a small improvement at a much higher cost. I also learned that meaningful comparisons require matching experimental settings and that different loss functions should not be compared as if they measured the same objective. My next step would be to complete the full ViT run and repeat the experiments with several seeds.

## References and result provenance

- [MNIST dataset specified in the assignment](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
- [Vision Transformer tutorial specified in the assignment](https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c)
- [Original Vision Transformer paper](https://arxiv.org/abs/2010.11929)
- [Torchvision documentation](https://docs.pytorch.org/vision/stable/index.html)
- Results and implementation reviewed from repository commit [`cc3d443`](https://github.com/saberBenhamda0/Deep_learning_-_Applications_assignments/tree/cc3d4439d3c0f6e5aa9a041a097e22e1f7af6f84/tp2).
