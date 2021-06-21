# Universal Domain Adaptation Under Label Noise
## Environment
Python 3.6.9, Pytorch 1.2.0, Torch Vision 0.4, Apex. See requirement.txt. We used apex from torch.cuda.

## Data Preparation

[Office](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/)  [OfficeHome](http://hemanthdv.org/OfficeHome-Dataset/)  [VisDA](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification)

Prepare dataset in data directory as follows.
```bash
./data/amazon/images/ ## Office
./data/Real/ ## OfficeHome
./data/visda/train/ ## VisDA synthetic images
./data/visda/val/ ## VisDA real images
```
Prepare image list.
```bash
unzip txt.zip
```
File list has to be stored in ./txt.

## Train

All training script is stored in script directory.

Example:

```bash
sh script/run_office_opda.sh $gpu-id configs/office-train-config_OPDA.yaml ##Office
sh script/run_officehome_opda.sh $gpu-id ##OfficeHome
sh script/run_visda.sh $gpu-id configs/visda-train-config_UDA.yaml ##VisDA
```
In each script, have to modify python file name accordingly.
```bash
train_dance_co.py --> For 100% of source samples in each batch for source loss
train_dance_coPer.py --> For some definite percentage of source samples in each batch for source loss
```
### Resutls:
Consider "per class mean acc" for our evaluation criteria.
```bash
Office(amazon2dslr), 45PN , stanrard pre-trained model, imagenet_loss factor=0 (can edit this in line 201 in train_dance_co.py)
[10000, [0.4166666666666667, 0.9047619047619048, 1.0, 0.0, 1.0, 0.7916666666666666, 0.7272727272727273, 0.6666666666666666, 0.375, 0.391304347826087, 0.8057142857142857], 'per class mean acc 0.643550296870455', 0.7289156626506024, 'closed acc 0.34036144578313254']

OfficeHome(Art2Clipart), 45PN , stanrard pre-trained model, imagenet_loss factor=0.25 (can edit this in line 201 in train_dance_co.py)
[6000, [0.65, 0.08928571428571429, 0.1875, 0.01020408163265306, 0.36363636363636365, 0.1111111111111111, 0.6164383561643836, 0.717391304347826, 0.5769230769230769, 0.18181818181818182, 0.20529393659587566], 'per class mean acc 0.337236556955926', 0.22680925142999253, 'closed acc 0.0954986321810495']

Office(dslr2webcam), 45SN , stanrard pre-trained model, imagenet_loss= 1 and percetange of source samples = 50% (can edit this in line 210 and 21 in train_dance_coPer.py)
[6000, [0.9655172413793104, 1.0, 0.9354838709677419, 0.9259259259259259, 0.18518518518518517, 0.9, 0.627906976744186, 0.6333333333333333, 0.7777777777777778, 0.26666666666666666, 0.8773234200743495], 'per class mean acc 0.7359200361867706', 0.7907801418439716, 'closed acc 0.3953900709219858']
```




