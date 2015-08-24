require 'cudnn'
local networks = require 'networks'

local use_cudnn = {}


-- Gives the number of output elements for a table of convolution layers
-- Also returns the new height (=width) of the image
local convs_noutput = function(convs, input_size)
  input_size = input_size or networks.base_input_size
  -- Get the number of channels for conv that are multiscale or not
  local nbr_input_channels = convs[1]:get(1).nInputPlane or
                             convs[1]:get(1):get(1).nInputPlane
  local output = torch.CudaTensor(nbr_input_channels, input_size, input_size)
  for _, conv in ipairs(convs) do
    conv:cuda()
    output = conv:forward(output)
  end
  return output:nElement(), output:size(2)
end

function use_cudnn.get_network(opt)
  -- Change the default modules to cudnn ones
  networks.modules.convolutionModule = cudnn.SpatialConvolution
  networks.modules.poolingModule = cudnn.SpatialMaxPooling
  networks.modules.nonLinearityModule = cudnn.ReLU

  -- Patch the convs_noutput method to use cuda
  networks.convs_noutput = convs_noutput

  -- Now just create the network using the original module
  local options = {}
  options.ms = opt.ms
  options.cnn = opt.cnn
  options.st = opt.st
  return networks.new(options)
end

return use_cudnn
