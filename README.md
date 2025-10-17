
---

### ✅ Polished Version of Your Current README (with small improvements)

```markdown
# Iris Tumor Detection using ResNet18 🧠🔬

### **About the Project**

Iris tumors are among the smallest cystic tumors associated with cancer, making detection particularly challenging due to their subtle size and appearance. If left untreated early, these tumors may lead to permanent blindness.

This project leverages **Transfer Learning and Fine-Tuning** with a **ResNet-18 architecture (pretrained on ImageNet)**.

* **Transfer Learning**: Utilizes pretrained ResNet-18 feature extractors to leverage knowledge from large-scale image datasets.
* **Fine-Tuning**: The final fully connected layer is replaced and trained specifically for **binary classification (tumor vs. no tumor)**. Dropout is applied to reduce overfitting.

Images are resized and normalized during preprocessing to optimize feature extraction. CNNs automatically capture subtle iris features that may indicate abnormalities. **Data augmentation techniques** were applied during training to improve generalization and robustness.

Integrating this technology into clinical workflows allows ophthalmologists to gain **real-time AI-assisted diagnostic support**, enabling early intervention and improved patient outcomes.

---

### **Datasets**

The dataset used for training and evaluation was collected from:

🔗 [The Rayid Method of Iris Analysis – Miles Research](http://milesresearch.com/main/links.htm)

The data is categorized into two groups:

* **Normal (No Tumor)**
* **Tumor Present**

Dataset access:

🔗 [Google Drive Link](https://drive.google.com/drive/folders/1JN3-8iQMFWO4FpGDQTa3QQLRhAoxVusJ)

---

### **Tech Stack**

* **PyTorch** – Training with Transfer Learning & Fine-Tuning (ResNet18)  
* **Streamlit** – Web application deployment  
* **Torchvision** – Pretrained models and transforms  
* **OpenCV & PIL** – Image preprocessing  

---

### **Usage**

1. Clone the repository:
   ```bash
   git clone <repo_url>
