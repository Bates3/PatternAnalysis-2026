# COMP3710 Report

**Benjamin Bateman**
48819101

---

## Section 3 — Feasability Check
### User Need, Scope & Acceptance Criteria
Doctors, more specifically those involved in MRI screening and diagnosing Alzheimer's disease, are the primary user for this prototype. False negatives are far more detrimental than false positives, so the prototype will sort MRI scans into three categories: confidently positive, doctor’s review, and confidently negative. With this in mind, the following acceptance criteria will be met:
- A minimum 0.8 test accuracy on AD vs CN (Alzheimer’s Disease vs Cognitively Normal)
- Less than 0.05 false negative score on classifying AD. Cases sent to specialists will count as predicting AD.
- Post-training classification of the entire test-set in under a minute (10,000 images).
- As the model will be trained locally on an AMD Radeon 7800XT gpu, VRAM usage must not exceed 14.5 GB.
- The model should be trained within a reasonable time (under 4 hours).

### Model Choice & Course Concepts
The ADNI dataset found on Rangpur has data already split into training and testing sets, as well as AD and NC. Images from both AD and NC classes will be combined together and labelled to be used for training.

ConvNeXt, a neural network achieved by applying choices which made the Swin Transformer successful onto ResNet-50, will be the primary model used for training. In choosing this model, I am choosing the hard difficulty band for the project. As ConvNeXt aims to be an improvement on ResNet-50, ResNet-50 will be used as a baseline model to compare to. 

To get confidence scores for the classifications, the sigmoid activation function will be used. This function squeezes outputs, so that those closer to 0 or 1 are more confident predictions.
#### Preliminary Feasibility Evidence
There are 10400 AD and 11120 NC training images and 4460 AD and 4540 NC testing images. All images are 256x240 pixels. All images are greyscale, meaning only 1 input channel. 
Resources were estimated using batches of size 8, 16, 32, 64, and 128 testing on the 27.8 million parameters of torchvision’s built-in model convnext_tiny with an AdamW optimiser run on three optimizer steps. This was run on the aforementioned AMD GPU.
bs=8: peak VRAM=1321 MiB
bs=16: peak VRAM=2119 MiB
bs=32: peak VRAM=3869 MiB
bs=64: peak VRAM=7200 MiB
bs=128: peak VRAM=13925 MiB
With these numbers, I will likely train on either a batch size of 64 or 128, testing both to achieve the best model the fastest.
### Risks, Budget & Fallback
Patient level data leakage is possible, where a patient has slices in both test and training sets, so an id check will need to be done to ensure this hasn’t occurred in the provided data. 
As I will be running the training locally, my budget is entirely related to the AMD GPU data shown above. The next planned experiment is to train ResNet-50 on the training data, recording VRAM usage and time. If I am unable to train locally due to AMD rocm bugs (some issues with batchnorm2d), I will fallback to training on Rangpur. If I am unable to figure out the ConvNeXt programming itself, I will switch to doing the improved 2D Improved U-NET HipMRI task instead.
