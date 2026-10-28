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
#### Preliminary Feasibi    lity Evidence
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

## ResNet-50:
best val auc 0.8476 @ epoch 27
---------------CUTOFF = 0.5 ---------------------
test: {'acc': np.float64(0.7579646017699115), 'ad_recall': np.float64(0.6767857142857143), 'nc_recall': np.float64(0.8377192982456141), 'auc': 0.824752702067669, 'fn_rate': np.float64(0.32321428571428573)}
test loss: 0.9186

Before updating training: {'acc': 0.7093, 'ad_recall': 0.7804, 'nc_recall': 0.6395, 'auc': 0.7876, 'fn_rate': 0.2196}

After updating trining: {'acc': 0.7586, 'ad_recall': 0.7384, 'nc_recall': 0.7785, 'auc': 0.8248, 'fn_rate': 0.2616}

## ConvNeXt
poch: 1 | train loss 0.6959889016690708 acc: 0.4993489583333333 | time: 157.44596481323242
{'acc': np.float64(0.4955357142857143), 'ad_recall': np.float64(1.0), 'nc_recall': np.float64(0.0), 'auc': 0.5092609423582876, 'fn_rate': np.float64(0.0)}
---------------CUTOFF = 0.5 ---------------------
epoch: 2 | train loss 0.6938918678178674 acc: 0.5113467261904762 | time: 313.89177918434143
{'acc': np.float64(0.5165178571428571), 'ad_recall': np.float64(0.6977477477477477), 'nc_recall': np.float64(0.33849557522123896), 'auc': 0.5304126803794946, 'fn_rate': np.float64(0.3022522522522523)}
---------------CUTOFF = 0.5 ---------------------
epoch: 3 | train loss 0.6941177325234527 acc: 0.5114862351190477 | time: 470.547616481781
{'acc': np.float64(0.5158482142857143), 'ad_recall': np.float64(0.08198198198198198), 'nc_recall': np.float64(0.9420353982300885), 'auc': 0.5451234752451567, 'fn_rate': np.float64(0.918018018018018)}
---------------CUTOFF = 0.5 ---------------------
epoch: 4 | train loss 0.7056000704566637 acc: 0.5056733630952381 | time: 627.708505153656
{'acc': np.float64(0.4955357142857143), 'ad_recall': np.float64(1.0), 'nc_recall': np.float64(0.0), 'auc': 0.5296995336044008, 'fn_rate': np.float64(0.0)}
---------------CUTOFF = 0.5 ---------------------
epoch: 5 | train loss 0.6988065104399409 acc: 0.5035342261904762 | time: 784.7627470493317
{'acc': np.float64(0.5044642857142857), 'ad_recall': np.float64(0.0), 'nc_recall': np.float64(1.0), 'auc': 0.5460158056286375, 'fn_rate': np.float64(1.0)}

1st proper training to 40 epochs it reached just 0.7 val accuracy and 0.7893 best auc
This transalated to a 0.68 test accuracy and 0.75 auc

Tried again with drop path = 0.1, 60 epochs and a learning rate back at 4e-3