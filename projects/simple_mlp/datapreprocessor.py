import torch
from sklearn.datasets import fetch_california_housing
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from torch.utils.data import Subset

def create_stratified_split(dataset, n_bins, n_splits):
    # Extract labels from dataset
    y = dataset.y.numpy()
    
    # Bin labels into discrete categories
    binned_y = np.digitize(y, bins=np.linspace(y.min(), y.max(), n_bins))
    
    # Initialize stratified split
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=0)
    
    # Get the first split (train/test) from the generator
    for train_index, val_index in sss.split(np.zeros(len(y)), binned_y):
        train_dataset = Subset(dataset, train_index)
        val_dataset = Subset(dataset, val_index)
        return train_dataset, val_dataset 
    
#You know how to do 
#inputation
#input normalization
#removal of outliers

#I am using the housing price prediction dataset, already cleaned and prepared (except for scaling)
X, y  = fetch_california_housing(return_X_y=True)

X = torch.as_tensor(X, dtype=torch.float32)
y = torch.as_tensor(y, dtype=torch.float32)

#It might be  a good idea to normalize the inputs and perhaps the targets as well, something like
# X = (X-X.mean(0))/X.std(0)
# y = (y-y.mean())/y.std()
#If you normalize the label, don't forget to unnormalize it in the prediction.
print(X.shape, y.shape)
print(y)

output_dir = Path("data")
output_dir.mkdir(exist_ok=True)

torch.save(X, "data/input_data.pt")
torch.save(y, "data/labels.pt")


