
### **About the Project**

Iris tumors are among the smallest cystic tumors associated with cancer, making their detection particularly challenging due to their subtle size and appearance. If left untreated in the early stages, these tumors may lead to permanent blindness.

To assist in diagnosis, this project leverages **Transfer Learning and Fine-Tuning** with a **ResNet-18 architecture (pretrained on ImageNet)**.

* **Transfer Learning**: Uses pretrained ResNet-18 feature extractors to take advantage of knowledge learned from large-scale image datasets.
* **Fine-Tuning**: The final fully connected layer is replaced and trained specifically for a **binary classification task (tumor vs. no tumor)**. Dropout is applied to reduce overfitting.

Images are resized and normalized during preprocessing to optimize feature extraction. CNNs enable the model to automatically capture subtle iris features that may indicate abnormalities. **Data augmentation techniques** were applied during training to improve generalization and robustness across varied image samples.

By integrating this technology into clinical workflows, ophthalmologists can gain **real-time AI-assisted diagnostic support**, enabling early intervention and improved patient outcomes.

---

### **Datasets**

The dataset used for training and evaluation was collected from:
🔗 [The Rayid Method of Iris Analysis – Miles Research](http://milesresearch.com/main/links.htm)

The data is categorized into two groups:

* **Normal (No Tumor)**
* **Tumor Present**

Dataset access:
📂
🔗 [Dataset-access]([http://milesresearch.com/main/links.htm](https://drive.google.com/drive/folders/1JN3-8iQMFWO4FpGDQTa3QQLRhAoxVusJ))
---

### **Tech Stack**

* **PyTorch** – Training with Transfer Learning & Fine-Tuning (ResNet18)
* **Streamlit** – Web application deployment
* **Torchvision** – Pretrained models and transforms
* **OpenCV & PIL** – Image preprocessing

---

🚀 Developed by **Naga Sai Chilukoti** as part of ongoing healthcare AI research.
