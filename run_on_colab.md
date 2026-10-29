# Running the thesis experiments on Google Colab (the exact steps used)

Runtime: **T4 GPU** (Runtime → Change runtime type → T4 GPU).
Results are written to Google Drive (`MyDrive/mona_v2`), so an interrupted session can be resumed: finished seeds are skipped.

## Cell 1 — data, Drive, libraries

```python
import kagglehub, glob, shutil, os
p = kagglehub.dataset_download("wcukierski/enron-email-dataset")
shutil.copy(glob.glob(p + "/**/emails.csv", recursive=True)[0], "/content/emails.csv")
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
!pip -q install shap lime datasketch
OUT = "/content/drive/MyDrive/mona_v2"
os.makedirs(OUT, exist_ok=True)
if os.path.exists("/content/thesis_full_code.py"):          # upload thesis_full_code.py once; it is then kept on Drive
    shutil.copy("/content/thesis_full_code.py", OUT + "/thesis_full_code.py")
!nvidia-smi --query-gpu=name --format=csv,noheader
```

## Cell 2 — experiments (about 5.5 hours on a T4)

```python
OUT = "/content/drive/MyDrive/mona_v2"
FILTER = 'grep --line-buffered -v -i -E "warning|cuda_timer|onednn|absl|^[IWE]0000"'
# 1) main condition with the ablation study (3 seeds)
!python -u {OUT}/thesis_full_code.py --emails /content/emails.csv --out {OUT} --conditions norm_random --main-seeds 42 43 44 --ablation 2>&1 | tee -a {OUT}/run_log.txt | {FILTER}
# 2) the other three conditions without ablation (norm_random is skipped from its checkpoints; final tables cover all four)
!python -u {OUT}/thesis_full_code.py --emails /content/emails.csv --out {OUT} --conditions raw_fixed raw_random norm_fixed norm_random --main-seeds 42 43 44 --other-seeds 42 43 44 2>&1 | tee -a {OUT}/run_log.txt | {FILTER}
```

The run is finished when the second `finished. Tables: ...` line appears.

## Outputs used in the thesis

| File (in `mona_v2/`) | Used for |
|---|---|
| `tables/Table_overall_accuracy.csv`, `Table_accuracy_by_length.csv` | Table 5.11 |
| `tables/Table_ablation_CAF_Net.csv` | Table 5.12 |
| `tables/Table_unseen_keys.csv` | Table 5.13 |
| `tables/Table_min_reliable_length.csv`, `Table_recall_by_cipher_and_length.csv` | text of Section 5.9 |
| `tables/Table_gate_char_branch.csv`, `Figure_CAF_gate_vs_length.png` | Figure 5.11 and the gate discussion |
| `norm_random/SHAP_RF_CAF.csv` | Figure 5.12 |
| `<condition>/Table_5_8_McNemar.csv` | McNemar results in Section 5.9 |

## Timing observed (T4, 66,780 messages → 267,120 ciphertexts per condition)

- Dataset build (cleaning, MinHash de-duplication, balancing): about 10 minutes per run.
- Attack-success features: about 12 minutes per 267,120 texts (cached as `cache_feat_v2_*.pkl`).
- One seed: 15–20 minutes without ablation, about 25 minutes with ablation.
