# EE782_FinalProject_VQA


---

##  Pipeline Phases

### **Phase 1 — Dataset Preprocessing**
- Loaded VQA v2 dataset  
- Tokenized questions using BERT  
- Generated soft-label answer vectors (3000-class)  
- Built `processed_samples` list  

### **Phase 2 — Baseline VisualBERT**
- Implemented a simplified VisualBERT  
- Verified training pipeline using dummy visual features  

### **Phase 3 — CLIP Vision Patch Extraction**
- Loaded CLIP ViT-B/32  
- Extracted **49×768 patch tokens** per image  
- Cached features for fast training  

### **Phase 4 — CLIP-VisualBERT Fusion Model**
- Added:
  - CLIP → BERT projection MLP  
  - Positional embeddings  
  - Modality embeddings  
  - Fusion via BERT encoder  

### **Phase 5 — Training & Saving**
- BCE loss with VQA soft targets  
- AdamW optimizer  
- Saved checkpoint:

  ### **Phase 8 — Gradio App**
Interactive web demo where users upload an image and ask any question.

---

##  Results

| Model | Accuracy |
|-------|----------|
| VisualBERT Baseline | ~0.20 |
| **CLIP-VisualBERT (ours)** | **0.34–0.36** |

---

## 🛠 Installation

```bash
pip install torch torchvision transformers gradio pillow tqdm
pip install git+https://github.com/openai/CLIP.git
