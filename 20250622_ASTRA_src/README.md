좋습니다. 요청하신 **VadCLIP 스타일의 README.md** 형식으로 바로 적용 가능한 ASTRA 버전을 정리해드리겠습니다.

---

# ASTRA

This is the official PyTorch implementation of our system:
**"ASTRA: Multimodal Weakly-Supervised Video Anomaly Detection Framework"**

![framework](data/framework.png)

---

## 🔍 Highlight

* We present **ASTRA**, a novel **multimodal weakly-supervised anomaly detection framework** that integrates **visual, audio, and text modalities** to automatically detect abnormal events in long video streams.
* With **Cross-Modal Attention (CMA)**, ASTRA selectively fuses complementary audio–visual cues, overcoming the limitations of unimodal detectors.
* A **Local–Global Temporal (LGT) module** leverages both Transformers and Graph Convolution Networks (Similarity & Distance graphs) to capture short- and long-range dependencies.
* A new **Contrastive Multi-modal Alignment Loss (CMAL)** based on InfoNCE aligns audio–visual representations under weak supervision, enabling precise anomaly localization without frame-level labels.
* **CLIP-based text prompt alignment** grounds anomaly categories (e.g., *fighting, shooting, explosion*) into the semantic space, providing **zero-shot adaptability** to unseen anomaly types.
* The system achieves robust **video-level classification** and **frame-level localization**, while significantly reducing annotation costs compared to fully-supervised methods.

---

## ⚙️ Environment Setup

We provide a Conda environment file for reproducibility:

```bash
conda env create -f environment.yml
conda activate astra
```

Key dependencies include:

* Python 3.10
* PyTorch 1.13.0, Torchvision 0.14.0, Torchaudio 0.13.0
* timm==0.6.7, einops, fvcore, scikit-learn, faiss-cpu, opencv-python
* transformers==4.31.0, accelerate, pytorchvideo

---

## 📊 Data Preparation

1. Extract **visual features** (512-d, CLIP/I3D) and **audio features** (128-d, VGGish).
2. Save them in `.npy` format.
3. Prepare CSV list files with feature paths and labels:

   * `list/xd_CLIP_rgb.csv`
   * `list/xd_CLIP_audio.csv`
   * `list/xd_CLIP_rgbtest.csv`
   * `list/xd_CLIP_audiotest.csv`

**Label mapping:**

```
A   : normal
B1  : fighting
B2  : shooting
B4  : riot
B5  : abuse
B6  : car accident
G   : explosion
```

---

## 🚀 Training and Testing

After setup, run the following commands:

**XD-Violence dataset**

```bash
python xd_train.py
python xd_test.py
```

**UCF-Crime dataset** (example extension)

```bash
python ucf_train.py
python ucf_test.py
```

---

## 📂 Project Structure

```
ASTRA/
├── model.py             # Main model (CLIPVAD, SingleModel)
├── layers.py            # Graph Convolution & Attention layers
├── CMA_MIL.py           # Contrastive Multi-modal Alignment Loss
├── InfoNCE.py           # InfoNCE contrastive loss
├── dataset.py           # Dataset loader (XDDataset, UCFDataset)
├── xd_train.py          # Training pipeline
├── xd_test.py           # Testing & evaluation
├── xd_option.py         # Argument parser & hyperparameters
├── xd_detectionMAP.py   # mAP computation module
├── tools.py             # Utility functions
├── environment.yml      # Conda environment
└── README.md
```

---

## 📚 References

We referenced the following repositories during implementation:

* [XDVioDet](https://github.com/Roc-Ng/XDVioDet)
* [DeepMIL](https://github.com/Roc-Ng/DeepMIL)

---

## 📜 Citation

If you find this repository useful for your research, please consider citing:

```bibtex
