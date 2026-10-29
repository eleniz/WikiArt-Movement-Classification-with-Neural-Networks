# WikiArt Movement Classification with Neural Networks

Pattern Recognition and Machine Learning Assignment.

This project studies image classification on a balanced WikiArt dataset containing paintings from four artistic movements. It compares a fully connected neural network, custom convolutional neural networks, and a frozen ImageNet-pretrained ResNet18 used as a linear probe.

## Project overview

The notebook:

- loads 5,400 RGB painting images stored in NumPy format;
- classifies images into Baroque, Impressionism, Cubism and Abstract Expressionism;
- normalises pixel values and converts images to PyTorch channel-first tensors;
- uses stratified train, validation and test splits;
- trains a feed-forward neural network on flattened pixels;
- incrementally develops custom CNN architectures;
- studies the effect of padding, pooling and ReLU activations;
- compares optimisers and CNN training configurations;
- evaluates data augmentation and regularisation;
- uses a frozen ImageNet-pretrained ResNet18 with a trainable four-class linear layer;
- compares accuracy, macro F1-score, parameter counts and training time;
- visualises confusion matrices and learned convolutional filters.

## Dataset

The supplied `wikiart_hw2.npz` file contains:

- 5,400 images;
- image shape `32 x 32 x 3`;
- `uint8` pixel values in the range `[0, 255]`;
- four balanced classes;
- 1,350 images per class.

| Class | Number of images |
|---|---:|
| Baroque | 1,350 |
| Impressionism | 1,350 |
| Cubism | 1,350 |
| Abstract Expressionism | 1,350 |

The notebook uses the following stratified split:

| Split | Total images | Images per class |
|---|---:|---:|
| Training | 3,200 | 800 |
| Validation | 800 | 200 |
| Test | 1,400 | 350 |

The dataset archive is included in this project as `wikiart_hw2.npz.zip`.

## Models

### Feed-forward neural network

The FFN receives flattened `32 x 32 x 3` pixel vectors. It has 397,604 trainable parameters and is trained with cross-entropy loss.

### Custom CNNs

The CNN experiments progressively compare:

1. convolutional layers without padding, pooling or activations;
2. convolutional layers with padding and pooling;
3. the complete architecture with ReLU activations;
4. a batch-normalised CNN with augmentation and regularisation experiments.

The main custom CNN has approximately 1.07 million trainable parameters.

Training-only augmentation consists of:

- random horizontal flips;
- random crops with padding;
- mild brightness and contrast jitter.

Validation and test images are not augmented.

### Frozen ResNet18

The final transfer-learning experiment uses an ImageNet-pretrained ResNet18. The backbone is frozen and only the final `512 -> 4` linear classification layer is trained.

- Total ResNet18 parameters: 11,689,512
- Trainable parameters: 2,052
- Training method: linear probe

## Results

The final held-out test-set comparison from the notebook is:

| Model | Test accuracy | Test macro F1 | Trainable parameters |
|---|---:|---:|---:|
| FFN on flattened pixels | 0.5057 | 0.5044 | 397,604 |
| Best custom CNN — augmentation only | 0.7079 | 0.7040 | 1,066,788 |
| Frozen ResNet18 + linear probe | **0.7614** | **0.7581** | **2,052** |

The frozen ResNet18 achieved the best result, outperforming the best custom CNN by `0.0542` macro F1, or approximately `7.7%` relative to the custom CNN's F1 score. This result shows the value of transferable visual representations, even when only a small classification head is trained on the target dataset.

### Augmentation and regularisation comparison

The notebook reports the following results for the batch-normalised CNN:

| Configuration | Best epoch | Validation macro F1 | Test macro F1 |
|---|---:|---:|---:|
| No augmentation, no regularisation | 27 | 0.6588 | 0.6481 |
| Augmentation only | 52 | 0.7250 | **0.7040** |
| Regularisation only | 25 | 0.7011 | 0.6606 |
| Augmentation + regularisation | 58 | **0.7260** | 0.6901 |

In this experiment, augmentation alone produced the strongest test performance. The combination of augmentation and regularisation achieved the highest validation F1 but did not achieve the highest test F1.

## Reproducibility

### Requirements

Use Python 3.10 or newer. Install the required packages:

```bash
pip install numpy matplotlib scikit-learn torch torchvision
```

For GPU execution, install the PyTorch build appropriate for your CUDA version from the [official PyTorch installation page](https://pytorch.org/get-started/locally/).

### Run in Google Colab

1. Open `wiki_classification_nn.ipynb` in Google Colab.
2. Upload `wikiart_hw2.npz.zip` to the Colab runtime.
3. Run the notebook from top to bottom.
4. Use a GPU runtime for the training sections when available.

The notebook uses seed `42` for NumPy, Python and PyTorch. Exact results can still vary across hardware and library versions, especially for GPU operations.

## Limitations

- Images are only `32 x 32`, which removes fine artistic details.
- The dataset contains only four movements and may not represent the full diversity of WikiArt.
- The external pretrained ResNet18 was trained on ImageNet rather than artwork.
- The ResNet experiment freezes the backbone, so the visual representation is not adapted to the target domain.
- Results are dependent on the supplied split and should not be treated as a general-purpose art historian or authentication system.

