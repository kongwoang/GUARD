# Data

GUARD runs on the saved outputs of each frozen model: predicted probabilities,
retrieval features and labels, for every degraded condition. These files are
model outputs, not the original datasets, and are hosted on Hugging Face.

```bash
bash data/download.sh              # all seven benchmarks, about 2.5 GB
bash data/download.sh mosei ave    # only some of them
```

Files go to `data/artifacts/`. Set `GUARD_ARTIFACTS` to use another folder.

| Benchmark | Frozen model | Folder |
|---|---|---|
| CMU-MOSEI | CMAD | `mosei_cmad/` |
| IEMOCAP | MoMKE | `iemocap_momke/` |
| OPPORTUNITY | DeepConvLSTM | `opportunity_dcl_v2/` |
| AVE | AV-att | `ave_av_att/` |
| NinaPro DB5 | sEMG CNN | `ninapro_cnn/` |
| PTB-XL | resnet1d_wang | `ptbxl_resnet1d_wang/` |
| DrugBAN (BioSNAP, BindingDB, Human) | DrugBAN | `drugban_processed/` |

The original datasets keep their own licences. IEMOCAP, NinaPro DB5, CMU-MOSEI
and AVE must be obtained from their authors to retrain the frozen models.
