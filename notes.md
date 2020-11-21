# Notes
## Data
* `V1-V28` - numeric variables obtained from PCA
* `Time` - seconds elapsed between transaction and the first transaction in the dataset
* `Amount` - the transaction amount
* `Class` - "genuine" or "fraud"

Note: `V1-V28` are private due to confidentiality reasons, so we can't do any data pruning.

## Process
1. Read in the data
2. Try oversampling vs. undersampling to balance dataset
3. Train-Test Split the data
4. Train different models and evaluate Accuracy & AUC
