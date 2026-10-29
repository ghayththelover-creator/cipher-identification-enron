# Cipher identification on Enron e-mail text

Code for the paper on identifying classical ciphers (Caesar, Vigenère, Playfair, Rail Fence) in
real e-mail text, comparing five baselines with RF-CAF and CAF-Net under four conditions
(raw_fixed, raw_random, norm_fixed, norm_random).

## Files
- `paper_code.py` – full experiment code (data preparation, ciphers, models, attack-success features, evaluation).
- `run_on_colab.md` – step-by-step instructions to run it on Google Colab (T4 GPU).
- `requirements.txt` – library versions.

## Data
The Enron e-mail corpus is downloaded from Kaggle (`wcukierski/enron-email-dataset`); it is not redistributed here.
Messages are cleaned, de-duplicated and balanced over three length strata (random_state = 42).

## Reproducing
Seeds 42, 43, 44. See `run_on_colab.md`.

## Citation
Add the paper reference and the Zenodo DOI after the first release.
