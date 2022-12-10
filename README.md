# Graph Learning

## Project structure
All code files are located in src. The main.py is a single entry point.
Src folder contains: 
- custom_layers.py required in tasks 3 and 4 
- data.py which provides all functions for loading and transforming data
- models.py which build model structure together
- validation.py provides interface for executing cross-validation. 

## Installation

If you like create and activate venv before.
```python
pip install -r requirements.txt
```
Please **unpack your dataset** in a folder above called datasets as in archive given in mooodle.
Thus, the path to it from this folder would be datasets `../datasets/NCI1/data.pkl`.

## Run
Python >= 3.8 is required.


You can run the GCN for different different datasets with cross-validation below.
```python
python -m src {NCI1 | Enzymes}
```

## Results

### Graph Level Neural Network
Iteration | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
Accuracy | XX | XX | XX | XX | XX | XX | XX | XX | XX | XX | XX 
--- --- --- --- --- --- --- 
The last 5 test accuracies from each of 10 folds:
68
67
74
73
75
72
72
62
74
70
69
72
72
75
73
73
72
75
71
73
73
73
62
71
71
70
69
70
73
70
67
70
69
71
73
66
70
67
71
73
66
73
76
75
76
78
75
75
74
72
69
75
68
63
76
74
70
73

mean: 71.36
std: 11.71

### Node Level Neural Network
Not implemented
