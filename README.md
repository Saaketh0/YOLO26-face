## Hey there, this is my code for finetuning the YOLO26n model on face data.

If you just want to download the final model, just go here:

https://platform.ultralytics.com/saaketh-sodanapalli/yolo26-face

### Lowkey much easier to train on ultralytics platform though, just need to:
- Download model with download_model.py
- Get YOLO dataset from change_data_format.py
- ZIP dataset up
- Upload all to Ultralytics


## Scores

* **mAP50:** 68.7%
* **mAP50-95:** 36.5%
* **Precision:** 84.6%
* **Recall:** 61.3%
* **F1:** 71.1%

### Credit for WIDER FACE dataset:
- All code done is for a noncommerical purpose
- Changes made to the dataset were formatting, so no derivatives were made

@inproceedings{yang2016wider,
Author = {Yang, Shuo and Luo, Ping and Loy, Chen Change and Tang, Xiaoou},
Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
Title = {WIDER FACE: A Face Detection Benchmark},
Year = {2016}}