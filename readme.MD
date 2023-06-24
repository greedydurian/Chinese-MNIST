## Environment
```conda create --name tractor```
```conda activate tractor```

## Requirements
Pillow==9.5.0
torch
torchvision
scikit-learn
matplotlib

pip3 install -r requirements.txt

Do note: No version control implemented

## Structure
```
assessment
├── data
│   └── data
│       └── images.jpg
├── dataProcessor
│   ├── dsLoader.py
│   ├── loader.py
│   └── transformData.py
├── misc
│   ├── model.drawio
│   └── reading.py
├── step1
│   ├── cnn.py
│   ├── training.py
│   └── __init__.py
├── step2
│   ├── metrics.py
│   ├── validation.py
│   └── __init__.py
├── chinese_mnist.csv
├── dataLoader.py
├── transformData.py
├── requirements.txt
├── model.pth
└── requirements.txt

```

1. dsLoader takes from 2 classes, transformData and loader
loader is the custom Pytorch used for loading and preprocessing of csv
transformData is a wrapper around loader 

2. step1 handles the training and model build 

3. step2 handles the validation and prediction of unseen data

4. Misc contains basic checks as well as the flow 
