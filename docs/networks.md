### Network Factory

This module allows to easily create networks either from command line options or a user defined module.

#### Basic use
`networks.new(opt)`

The `opt` field contains the parameters to build the network:
* `opt.cnn` will create a network with the same depth as the one in LeCun's [paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf): two convolutions and two fully connected layers. It contains a string of 3 comma separated values, the two first values will be the number of output channels for the convolutions, the last one the number of hidden units between the two linear layers. The generated network will use 5x5 kernels for the convolutions and 2x2 kernels for the pooling.
* `opt.ms` to enable the multiscale mode. This will use both the outputs of the first and second convolution as input to the fully connected layer.
* `opt.no_cnorm` to disable the contrastive normalization on the convolutional layers.

#### Advanced use

For advanced use, you can create your own module that will create the network. You can provide the module with the `opt.net` option to the factory.
Your module should return a table containing a function `get_network(opt)` that will be called with the original options from the program.
Your module can use the following elements from the networks module:
* `networks.modules` are the basic modules, they can be changed by the user (to use cudnn version for example). It contains:
    * `networks.modules.convolutionModule` the convolution module (default nn.SpatialConvolutionMM)
    * `networks.modules.poolingModule` the pooling module (default nn.SpatialMaxPooling)
    * `networks.modules.nonLinearityModule` the non linearity module (default nn.ReLU)

* `networks.new_conv(ic, oc, ms, no_cnorm, ks)` will create a convolution module: conv + nonLinearity + pooling (+ contrastiveNormalization)
    * `ic`: number of input channels for the convolution
    * `oc`: number of output channels for the convolution
    * `ms`: use or not of multiscale (default false)
    * `no_cnorm`: do not use the contrastive normalization (default false)
    * `ks`: size of convolution kernels (default 5)

* `networks.convs_noutput(convs, is)` will return the number of elements in the output after forwarding an image of height (=width) `is` through all the convolutions given in `convs`.

* `netowrks.new_fc(is, os)` Create a new fully connected layer: view + linear + nonLinearity
    * `is`: number of input nodes
    * `os`: number of output nodes

* `networks.new_classifier(is, os)` Create a new classifier layer: view + linear
    * `is`: number of input nodes
    * `os`: number of output nodes

* `networks.new_spatial_transformer(locnet, rot, sca, tra, is, ic)` will create a spatial transformer module (see [here](https://github.com/qassemoquab/stnbhwd/issues/1#issue-95428051) for the structure)
    * `locnet`: options allowing to generate the localization network. They are formatted the same way as `opt.cnn`
    * `rot, sca, tra`: options to restrict the allowed transformations. If none of them is true, a fully parametrized version is used.
    * `is`: the height (=width) of the input image
    * `ic`: the number of channels in the input image


### Example

This example uses the following user defined modules:
* [`use_cudnn.lua`](../use_cudnn.lua) that allows to use [cudnn binding](https://github.com/soumith/cudnn.torch) for faster runtime
* [`idsua_net.lua`](../idsia_net.lua) that allows to use network with the shape of the winner of the gtsrb competition: [idsia paper](http://people.idsia.ch/~juergen/nn2012traffic.pdf)

```lua
local networks = require 'networks'

-- Basic network
local opt = {}
opt.cnn = "108,200,100"
local network = networks.new(opt)

-- Basic network using cudnn backend
local opt = {}
opt.cnn = "108,200,100"
opt.net = "use_cudnn.lua"
local network = networks.new(opt)

-- Idsia network
local opt = {}
opt.cnn = "150,200,300,350" -- idsia network needs 4 parameters
opt.net = "idsia_net.lua"
local network = networks.new(opt)

-- spatial transformer network
local opt={}
opt.cnn = "108,200,100"
opt.st = true
opt.locnet = "30,60,30" -- small localization network
local network = networks.new(opt)
```
