import torch
from sklearn.datasets import fetch_california_housing

#You know how to do 
#inputation
#input normalization
#removal of outliers

#I am using the housing price prediction dataset, already cleaned and prepared
X, y  = fetch_california_housing(return_X_y=True)

torch.as_tensor(X, dtype=torch.float32)
torch.as_tensor(y, dtype=torch.float32)
print(X.shape, y.shape)
print(y)
torch.save(X, "data/input_data.pt")
torch.save(y, "data/labels.pt")
