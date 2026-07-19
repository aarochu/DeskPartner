# P5 structure

```
p5_training/
├── GO_NO_GO.md
├── README.md
├── BAKEOFF.md
├── build_dataset_mixture.py   # from verified LeRobot ds → mixture yaml
├── modal_finetune.py          # Modal LoRA job + wall-clock timeout
├── eval_checkpoint.py         # N trials → CSV
├── bakeoff.py                 # scripted vs policy → VERDICT line
└── configs/
    ├── molmoact2_single_arm.yaml
    └── mixture_deskpartner.yaml   # generated
```
