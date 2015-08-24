### Trainer

This module allows to perform training from a given network, dataset and options.

#### Initialization
`trainer.initialize(network, criterion, opt)`

This functions initializes the basic states of the optimizer.
It must be called before any other method on the trainer.
It takes as input the `network`, the `criterion` and an option table.
The possible options are:
* `opt.lr`: the learning rate
* `opt.lrd`: the learning rate decay
* `opt.mom`: the momentum
* `opt.wd`: the weight decay
* `opt.no_cuda`: to disable the use of cuda
* `opt.bs`: the minibatch size

#### Training
`trainer.train(dataset)`

This function performs one epoch of training using the given dataset.
It returns the total error on the dataset.

#### Testing
`trainer.test(dataset)`

This function evaluates the current network on the given dataset.
It returns both the total error and accuracy on the dataset.


### Example

```lua
local trainer = require 'trainer'

local train_dataset, test_dataset, network, criterion
------
-- Initialize all these elements with the other modules

local opt = {}
opt.lr = 0.01

trainer.initialize(network, criterion, opt)

local accuracy
for epoch=1,10 do
  trainer.train(train_dataset)
  _, accuracy = trainer.test(test_dataset)
end

print("Final accuracy: "..accuracy)
```
