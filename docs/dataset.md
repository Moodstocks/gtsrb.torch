### Data Loader module

The data loader is contained in the `dataset.lua` module.
This module allows to download and prepare the gtsrb dataset for use with torch.

#### Training dataset
`dataset.get_train_dataset(nbr_examples, use_validation)`

This function returns a Lua table containing two fields:
* `data` contains a torch tensor of size (nbr_examples x 3 x 48 x 48) containing the train images
* `label` contains a torch tensor of size (nbr_examples x 1) containing the train labels

If the argument `nbr_examples` is not specified, it returns the full training dataset containing 39,209 examples.
If the argument `use_validation` is `true`, this function will return two tables, one with the train set and one with the validation set. The split is done following LeCun's [paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) methodology. In this case, the full training set contains 37,919 examples and the validation set contains 1,290 examples.

#### Testing dataset
`dataset.get_test_dataset(nbr_examples)`

This function returns a Lua table containing two fields:
* `data` contains a torch tensor of (size nbr_examples x 3 x 48 x 48) containing the test images
* `label` contains a torch tensor of (size nbr_examples x 1) containing the test labels

If the argument nbr_examples is not specified, it returns the full testing dataset containing 12,630 examples.

**Warning** If the number of example is not limited, the data will be ordered by label (you may need to shuffle them before training). If the number of example is limited, the returned data will be shuffled.

#### Global normalization
`dataset.normalize_global(dataset, mean, std)`

This function performs global normalization on the given dataset.
If `mean` and `std` are not specified, they are computed on the given dataset.
It returns the `mean` and `std` values used to normalize the dataset.

#### Local normalization
`dataset.normalize_local(dataset)`

This function performs local normalization on the given dataset.


### Example

```lua
local dataset = require 'dataset'

-- Get 20,000 training examples
local train_dataset = dataset.get_train_dataset(20000)
-- Get the whole testing dataset
local test_dataset = dataset.get_test_dataset()

-- performs global normalization on both datasets
local mean, std = dataset.normalize_global(train_dataset)
dataset.normalize_global(test_dataset, mean, std)

-- perform local normalization
dataset.normalize_local(train_dataset)
dataset.normalize_local(test_dataset)
```
