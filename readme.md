# ACFL

The codes for the work "ACFL: Communication-Efficient adversarial contrastive federated learning for medical image segmentation". Our paper has been accepted by Knowledge-Based Systems. We updated the Reproducibility. I hope this will help you to reproduce the results. 



## Requirements

- torch 1.11.0
- torchvision 0.12.0

## Dataset

Firstly create directory for log files and change the dataset path (`pacs_path`, `officehome_path` and `terrainc_path`) and log path (`log_count_path`) in configs/default.py.
Please download the datasets from the official links:

- [PACS](http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017)
- [OfficeHome](https://hemanthdv.github.io/officehome-dataset)
- [TerraInc](https://beerys.github.io/CaltechCameraTraps)


## Train/Test
When you want to train, you run the .sh file. The corresponding background running script has been configured in the.sh file. You can modify the --dataset parameter to select the corresponding dataset.

- Train/Test
```
sh run.sh
```


## Acknowledgement

This project has benefited from the following resources, and I would like to express my gratitude:

- FACT [https://github.com/MediaBrain-SJTU/FACT]
- DomainBed [https://github.com/facebookresearch/DomainBed]
- FedNova [https://github.com/JYWa/FedNova]
- SCAFFOLD-PyTorch [https://github.com/KarhouTam/SCAFFOLD-PyTorch]



## Citation

```latex
@article{liang2024acfl,
  title={ACFL: Communication-Efficient adversarial contrastive federated learning for medical image segmentation},
  author={Liang, Zhiwei and Zhao, Kui and Liang, Gang and Wu, Yifei and Guo, Jinxi},
  journal={Knowledge-Based Systems},
  volume={304},
  pages={112516},
  year={2024},
  publisher={Elsevier}
}
```

