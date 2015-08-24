require 'torch'
require 'image'
require 'cunn'
require 'cudnn'
require 'stn'
local gtsrb = require 'gtsrb'
local lapp = require 'pl.lapp'

local opt = lapp [[
Plotting script
Required option
  -i, --input   (string)      Input model

Optional parameters
  -n            (default 50)  Use only N samples for plot
  -t,--train                  Plot data from the train set and not test one
  --no_pt                     Do not print the limit points in the original image
  --no_lnorm                  Do not locally normalize the data
  -s,--seed     (default 1)   Random seed value (-1 to disable)
]]

if opt.seed > 0 then
  torch.manualSeed(opt.seed)
  math.randomseed(opt.seed)
end

print("Loading network...")
local network = torch.load(opt.input)
local mean_std = torch.load(opt.input..'norm')

print("Loading data...")
local to_plot_dataset = gtsrb.dataset.get_test_dataset(opt.n)
print("Global normalization of the dataset...")
gtsrb.dataset.normalize_global(to_plot_dataset, mean_std[1], mean_std[2])
if not opt.no_lnorm then
  print("Local normalization of the dataset...")
  gtsrb.dataset.normalize_local(to_plot_dataset)
end

print('Extracting features...')
if not opt.no_cuda then
  to_plot_dataset.data = to_plot_dataset.data:cuda()
  network = network:cuda()
end
local scores = network:forward(to_plot_dataset.data)
local st_output = network:get(1).output

if not opt.no_pt then
  print("Adding limit points on the original images...")
  -- Get the transformation matrix from the AffineTransformMatrixGenerator
  local transfo = network:get(1):get(1):get(2):get(2).output:float()
  local corners = torch.Tensor{{-1,-1,1},{-1,1,1},{1,-1,1},{1,1,1}}
  -- Compute the positions of the corners in the original image
  local points = torch.bmm(corners:repeatTensor(opt.n,1,1), transfo:transpose(2,3))
  -- Ensure these points are still in the image
  points = torch.floor((points+1)*48/2)
  points:clamp(1,47)
  for batch=1,opt.n do
    for pt=1,4 do
      local point = points[batch][pt]
      for chan=1,3 do
        local max_value = to_plot_dataset.data[batch][chan]:max()*1.1
        -- We add 4 white pixels because one can disappear in image rescaling
        to_plot_dataset.data[batch][chan][point[1]][point[2]] = max_value
        to_plot_dataset.data[batch][chan][point[1]+1][point[2]] = max_value
        to_plot_dataset.data[batch][chan][point[1]][point[2]+1] = max_value
        to_plot_dataset.data[batch][chan][point[1]+1][point[2]+1] = max_value
      end
    end
  end
end

print("Plotting the results...")
image.display{image=to_plot_dataset.data, padding=2}
image.display{image=st_output, padding=2}

