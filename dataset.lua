local pl = (require 'pl.import_into')()

require 'torch'
require 'image'

torch.setdefaulttensortype('torch.FloatTensor')

local dataset = {}

-- Private function declaration
local generate_dataset
local prune_dataset

-- These paths should not be changed
dataset.path_remote_train = "http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip"
dataset.path_remote_test = "http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip"
dataset.path_remote_test_gt = "http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_GT.zip"

dataset.train_dataset_bin = "train_dataset.bin"
dataset.test_dataset_bin = "test_dataset.bin"


-- This function will download the dataset in the './GTSRB' temp folder, and generate
-- binary files containing the dataset as torch tensors.
function dataset.download_generate_bin()
  if not pl.path.isfile(dataset.train_dataset_bin) or
     not pl.path.isfile(dataset.test_dataset_bin) then

     if not pl.path.isdir('GTSRB') then
      local tar_train = paths.basename(dataset.path_remote_train)
      local tar_test = paths.basename(dataset.path_remote_test)
      local tar_test_gt = paths.basename(dataset.path_remote_test_gt)

      print('Downloading dataset...')
      os.execute('wget ' .. dataset.path_remote_train .. '; ' ..
                 'unzip ' .. tar_train .. '; '..
                 'rm ' .. tar_train)
      os.execute('wget ' .. dataset.path_remote_test .. '; ' ..
                 'unzip ' .. tar_test .. '; '..
                 'rm ' .. tar_test .. '; ' ..
                 'mkdir GTSRB/Final_Test/Images/final_test; ' ..
                 -- too many arguments for a plain mv...
                 [[find GTSRB/Final_Test/Images/ -maxdepth 1 -name '*.ppm' -exec sh -c 'mv "$@" "$0"' GTSRB/Final_Test/Images/final_test/ {} +;]] ..
                 'rm GTSRB/Final_Test/Images/GT-final_test.test.csv')
      os.execute('wget ' .. dataset.path_remote_test_gt .. '; ' ..
                 'unzip ' .. tar_test_gt .. '; '..
                 'rm ' .. tar_test_gt .. '; ' ..
                 'mv GT-final_test.csv GTSRB/Final_Test/Images/final_test/GT-final_test.csv')
    end

    print('Generating bin of the dataset')
    local train_set = generate_dataset('GTSRB/Final_Training/Images')
    torch.save('train_dataset.bin', train_set)
    train_set = nil
    collectgarbage()
    local test_set = generate_dataset('GTSRB/Final_Test/Images')
    torch.save('test_dataset.bin', test_set)
    test_set = nil
    collectgarbage()

    if paths.dirp('GTSRB') then
      os.execute('rm -r GTSRB')
    end
  end
end

-------------------------------------------------
-- Main Interface
-------------------------------------------------

-- Returns the train dataset
-- nbr_examples is optional and allows to get only a subset of the training samples
-- Warning: if the number of examples is not limited, the dataset is ordered by class
-- If the number of examples is limited, the subset will be shuffled.
function dataset.get_train_dataset(nbr_examples)
  dataset.download_generate_bin()
  local train_dataset = torch.load(dataset.train_dataset_bin)

  -- Limit the number of samples if required by the user
  if nbr_examples and nbr_examples ~= -1 then
    train_dataset = prune_dataset(train_dataset, nbr_examples)
  end
  return train_dataset
end

-- Returns the test dataset
-- nbr_examples is optional and allows to get only a subset of the testing samples
-- Warning: if the number of examples is not limited, the dataset is ordered by class
-- If the number of examples is limited, the subset will be shuffled.
function dataset.get_test_dataset(nbr_examples)
  dataset.download_generate_bin()
  local test_dataset = torch.load(dataset.test_dataset_bin)

  -- Limit the number of samples if required by the user
  if nbr_examples and nbr_examples ~= -1 then
    test_dataset = prune_dataset(test_dataset, nbr_examples)
  end
  return test_dataset
end

-- Normalize the given dataset
-- You can specify the mean and std values, otherwise, they are computed on the given dataset
-- Return the mean and std values
function dataset.normalize_global(dataset, mean, std)
  local std = std or dataset.data:std()
  local mean = mean or dataset.data:mean()
  dataset.data:add(-mean)
  dataset.data:div(std)
  return mean, std
end

-- Locally normalize the dataset
function dataset.normalize_local(dataset)
  require 'image'
  local norm_kernel = image.gaussian1D(7)
  local norm = nn.SpatialContrastiveNormalization(3,norm_kernel)
  local batch = 200 -- Can be reduced if you experience memory issues
  local dataset_size = dataset.data:size(1)
  for i=1,dataset_size,batch do
    local local_batch = math.min(dataset_size,i+batch) - i
    local normalized_images = norm:forward(dataset.data:narrow(1,i,local_batch))
    dataset.data:narrow(1,i,local_batch):copy(normalized_images)
  end
end

-------------------------------------------------
-- Private function
-------------------------------------------------

prune_dataset = function(dataset, nbr_examples)
  -- Limit the number of samples if required by the user
  assert(nbr_examples and nbr_examples > 1 and nbr_examples < dataset.data:size(1),
         'Invalid number of examples required, not within dataset range.')

  local randperm = torch.randperm(dataset.data:size(1))
  local subset_data = torch.Tensor(nbr_examples, 3, 48, 48)
  local subset_label = torch.Tensor(nbr_examples, 1)
  for i=1,nbr_examples do
    subset_data[i]:copy(dataset.data[randperm[i]])
    subset_label[i]:copy(dataset.label[randperm[i]])
  end
  dataset.data = subset_data
  dataset.label = subset_label
  collectgarbage()

  return dataset
end

-- This will generate a dataset as torch tensor from a directory of images
-- parent_path is a string of the path containing all the images
generate_dataset = function(parent_path)
  assert(parent_path, "A parent path is needed to generate the dataset")

  local main_dataset = {}
  main_dataset.nbr_elements = 0

  local images_directories = pl.dir.getdirectories(parent_path)
  table.sort(images_directories)

  for image_directory_id, image_directory in ipairs(images_directories) do
    local csv_file_name = 'GT-' .. pl.path.basename(image_directory) .. '.csv'
    local csv_file_path = pl.path.join(image_directory, csv_file_name)

    local csv_content = pl.data.read(csv_file_path)

    local filename_index = csv_content.fieldnames:index('Filename')
    local class_id_index = csv_content.fieldnames:index('ClassId')

    for image_index, image_metadata in ipairs(csv_content) do
      local image_path = pl.path.join(image_directory, image_metadata[filename_index])
      local image_data = image.load(image_path)

      -- We do no transformation but a rescaling so all the images have the same size
      image_data = image.scale(image_data, 48, 48)

      local label = torch.Tensor{image_metadata[class_id_index]+1}

      main_dataset.nbr_elements = main_dataset.nbr_elements + 1
      main_dataset[main_dataset.nbr_elements] = {image_data, label}

      if image_index % 50 == 0 then
        collectgarbage()
      end
    end
  end

  -- Store everything as proper torch Tensor now that we know the total size
  local main_data = torch.Tensor(main_dataset.nbr_elements, 3, 48, 48)
  local main_label = torch.Tensor(main_dataset.nbr_elements, 1)
  for i,pair in ipairs(main_dataset) do
    main_data[i]:copy(main_dataset[i][1])
    main_label[i]:copy(main_dataset[i][2])
  end
  main_dataset = {}
  main_dataset.data = main_data
  main_dataset.label = main_label

  return main_dataset
end

-- Return the module
return dataset
