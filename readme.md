# MIGGRI

This repository contains the codes for ''MIGGRI: a multi-instance graph neural network model for inferring gene regulatory network from spatial expression images''

![](https://github.com/YYHemich/MIGGRI/edit/main/modelStage1_refined.png)

![](https://github.com/YYHemich/MIGGRI/edit/main/modelStage2-refined.png)



We provides the extracted features in data/con_128.pkl.

The gene regulatory interactions is in data/regulatory_relationship.pkl



### Usage

Training the model with validation by

   	python train.py --bidirectional

To retrain the model adding validation data into training with the best model setting on validation, execute

```
python retrain.py --model "src of the model"
```

A running example is given in main.sh.

To evaluate the model performance, execute

```
python test.py --model "scr of the model"
```

