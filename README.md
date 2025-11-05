# Glyphic-Semantic Siamese BiLSTM (GSSBL)

This repository implements a **Glyphic-Semantic Siamese BiLSTM (GSSBL)** that integrates **semantic** (meaning) and **glyphic** (shape) embeddings of oracle bone inscriptions (OBIs) to address the **Fragment Association Prediction Problem**.
This model provides a new AI-based approach for **oracle bone rejoin**. It was proposed in the paper *"A Multi-Modal Dataset and Method for Bone-Level Association Prediction in Oracle Bone Inscriptions"*.


---

## ✨ Features
- Dual-tower BiLSTM encoders for two sentences.
- Learnable fusion weight `α` to combine **meaning/semantic** and **shape/glyphic** embeddings.
- Switch between **primary-character** and **secondary-character** tags via `use_subword`.
- Early stopping on training loss (patience).
- Metrics: **AUROC** , **AUPR** , **Accuracy** , **Precision** , **Recall** and **F1 score**.
- Multiple runs with averaged results saved to CSV.

---

## 📦 Requirements & Installation

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
```

Minimal `requirements.txt`:
```txt
numpy
torch
scikit-learn
tqdm
Pillow
```

Python 3.9+ and PyTorch ≥ 2.0 are recommended.

---

## 📂 Data Format

Training and test files (`train.txt`, `test.txt`) should be **tab-separated** with three fields per line:

```
sentence1 \t sentence2 \t label
```

Each sentence is a whitespace-separated sequence of tokens in the form `primary-character|secondary-character`, e.g.:

```
u7v6rlhp81|xzku05tqy7 h0gzv3styy|h0gzv3styy jvuf4ut3c5|bf98fqv8qx	d7mczw6osp|d7mczw6osp 7h3wu2xyyf|7h3wu2xyyf 6ceuhy4fvr|6ceuhy4fvr	1
```

- `label = 1` → the two OBIs are from the **same original oracle bone**  
- `label = 0` → the two OBIs are from **different oracle bones**


**Sequence length** is truncated to 30 tokens by default in the script.

---

## 🔤 Pretrained Embeddings

Provide two JSON files of the same vocabulary (keys are tokens, values are 512-d vectors by default):

- `model/sgns.json` — semantic embeddings
- `model/glyph.json` — glyphic embeddings

Example format:

```txt
{
  "8gxzzbv7w8": [0.1, 0.2, 0.3, /* ... 512 dims ... */],
  "jrzjjh3g1r": [0.05, 0.07, 0.11 /* ... */],
  ...
}
```

> The semantic embeddings are learned using the skip-gram model with negative sampling, which can be trained via `w2v.py`.

> The data required for training the semantic embeddings can be accessed at [Zenodo](https://zenodo.org/records/14882488). The processed version of the data is also provided in the `data` folder of this GitHub repository.

> The glyphic embeddings can be trained using `vae.py`. The data required for this step can be accessed at [Zenodo](https://zenodo.org/records/14882488).

> To optimise glyphic embeddings using the secondary-character glyphic contrastive learning module (SGCLM), please use `contrastive.py`.

> The file **`data/tag.txt`** stores the **primary–secondary tag relationships** used in the SGCLM.  Each line represents a single *primary-character tag* followed by its corresponding *secondary-character tags*.  All secondary-character tags listed in the same line belong to the same primary-character tag.


---

## 🚀 Usage

### Option A: main.py

We provide a `main.py`. You can modify the parameters inside before running it.

### Option B: CLI wrapper (optional)


```bash
python code/GSSBL.py   --train_path data/train.txt   --test_path data/test.txt   --word_vec_path_meaning model/sgns.json   --word_vec_path_shape model/glyph.json   --output_csv_path results/output.csv   --repeat 5   --use_subword True   --cuda_device 0
```


> If `subchara = False`, keys should correspond to **primary-character** tokens (the right side of `primary-character|secondary-character`). Otherwise, keys correspond to **secondary-character** (the left side).

## ⚙️ Arguments

- `train_path` / `test_path`: Paths to TSV files with sentence pairs and labels.
- `word_vec_path_meaning`: Path to semantic embedding JSON.
- `word_vec_path_shape`: Path to glyphic embedding JSON.
- `output_csv_path`: Where to save averaged metrics.
- `repeat`: Number of repeated runs (default in examples: 5).
- `use_subword`: Use secondary-character tokens (`True`) or primary-characters (`False`).
- `cuda_device`: GPU index; CPU used if CUDA is unavailable.

---

## 📊 Output

- Console logs each run’s **AUROC** , **AUPR** , **Accuracy** , **Precision** , **Recall** and **F1 score**.
- CSV file with averaged metrics:
  ```
  Run,AUROC,AUPR,ACC,Precision,Recall,F1
  1,0.9572285274870893,0.7864667610863505,0.9520656314093173,0.7449933244325768,0.7190721649484536,0.7318032786885246
  ```

---

## 🧠 Notes on Training

- Early stopping is based on **training loss** with a patience window (default 10).
- Embedding dimension (`EMBED_DIM=512`), hidden size (`HIDDEN_SIZE=256`), dropout (`0.3`), batch size (`64`), learning rate (`5e-4`) and max epochs (`100`) are set inside the function.
- Sequences are truncated to `MAX_SEQ_LEN=30` tokens.
- Vocabulary is built from **both** train and test files to avoid OOV at test time.
- Reproducibility: random seeds are **not** set by default.

---

## 📁 Suggested Repository Structure

```                
├── requirements.txt
├── data/
│   ├── processed_data.txt
│   ├── train.txt
│   └── test.txt
├── model/
│   ├── sgns.json
│   └── glyph.json
├── code/
│   ├── bilstm.py
│   ├── BoW.py
│   ├── constractive.py
│   ├── GSSBL.py
│   ├── jvtrans.py
│   ├── main.py
│   ├── SIF.py
│   ├── te.py
│   ├── TextCNN.py
│   ├── uSIF.py
│   ├── vae.py
│   └── w2v.py
└── results/
    └── output.csv
```

---



## 📖 Citation

If you use this code in your research, please cite:

```bibtex
not available yet
```
