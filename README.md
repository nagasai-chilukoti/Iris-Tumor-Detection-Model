🌐 **Project Website:**
[https://nagasai-chilukoti-iris-tumor-detection.streamlit.app](https://nagasai-chilukoti-iris-tumor-detection.streamlit.app)https://iris-tumors-detector.streamlit.app/
This model is deployed using **Streamlit**, an open-source deployment framework widely used for interactive data apps. The application runs on Streamlit Cloud and provides a clean, intuitive interface for healthcare researchers and professionals. Users can upload an iris image, and the system processes it to determine whether a tumor is present. The interface includes image previews, probability breakdowns, and clear predictions for quick interpretation.

---

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
📂 [Google Drive – Iris Tumor Dataset]([https://drive.google.com/drive/folders/1Tzc9ym41ni1K9g9zDck3tRQ7i5MaWDoS?usp=drive_link](https://drive.google.com/drive/folders/1JN3-8iQMFWO4FpGDQTa3QQLRhAoxVusJ?q=sharedwith:public%20parent:1JN3-8iQMFWO4FpGDQTa3QQLRhAoxVusJ))

---

### **Tech Stack**

* **PyTorch** – Training with Transfer Learning & Fine-Tuning (ResNet18)
* **Streamlit** – Web application deployment
* **Torchvision** – Pretrained models and transforms
* **OpenCV & PIL** – Image preprocessing

---

🚀 Developed by **Naga Sai Chilukoti** as part of ongoing healthcare AI research.
