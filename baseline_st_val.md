## Results
This file contains the results for the tests on the GTSRB dataset.

| Command line arguments | accuracy on validation set |
| --------------------- | ----- |
|  |  |
| --val --net idsia_net.lua --cnn 100,150,250,300 --st  --locnet 100,200,100 | 0.99147286821705 |
| --val --net idsia_net.lua --cnn 150,200,300,350 --st  --locnet 100,200,100 | 0.99147286821705 |
| --val --net idsia_net.lua --cnn 100,150,250,300 --st --sca --tra --locnet 100,200,100 | 0.98992248062016 |
| --val --net idsia_net.lua --cnn 150,200,300,350 --st --sca --tra --locnet 100,200,100 | 0.9922480620155 |
| --val --net idsia_net.lua --cnn 100,150,250,300 --st  --locnet 150,250,150 | 0.99147286821705 |
| --val --net idsia_net.lua --cnn 150,200,300,350 --st  --locnet 150,250,150 | 0.99147286821705 |
| --val --net idsia_net.lua --cnn 100,150,250,300 --st --sca --tra --locnet 150,250,150 | 0.98992248062016 |
| --val --net idsia_net.lua --cnn 150,200,300,350 --st --sca --tra --locnet 150,250,150 | 0.99147286821705 |
|  |  |
| --val --st --net idsia_net.lua --cnn 150,200,300,350 --locnet ""   | 0.97829457364341 |
| --val --st --net idsia_net.lua --cnn 200,250,350,400 --locnet ""   | 0.97674418604651 |
| --val --st --net idsia_net.lua --cnn 150,200,300,350 --locnet 150,150,150   | 0.9937984496124 |
| --val --st --net idsia_net.lua --cnn 200,250,350,400 --locnet 150,150,150   | 0.9922480620155 |
| --val --st --net idsia_net.lua --cnn 150,200,300,350 --locnet 200,300,200   | 0.98992248062016 |
| --val --st --net idsia_net.lua --cnn 200,250,350,400 --locnet 200,300,200   | 0.99147286821705 |
| --val --st --net idsia_net.lua --cnn 150,200,300,350 --locnet "" --locnet2 150,150,150  | 0.99147286821705 |
| --val --st --net idsia_net.lua --cnn 200,250,350,400 --locnet "" --locnet2 150,150,150  | 0.9937984496124 |
| --val --st --net idsia_net.lua --cnn 150,200,300,350 --locnet 150,150,150 --locnet2 150,150,150  | 0.99302325581395 |
| --val --st --net idsia_net.lua --cnn 200,250,350,400 --locnet 150,150,150 --locnet2 150,150,150  | 0.9922480620155 |
| --val --st --net idsia_net.lua --cnn 150,200,300,350 --locnet 200,300,200 --locnet2 150,150,150  | 0.9937984496124 |
| --val --st --net idsia_net.lua --cnn 200,250,350,400 --locnet 200,300,200 --locnet2 150,150,150  | 0.9922480620155 |
| --val --st --net idsia_net.lua --cnn 150,200,300,350 --locnet "" --locnet2 75,75,75  | 0.98294573643411 |
| --val --st --net idsia_net.lua --cnn 200,250,350,400 --locnet "" --locnet2 75,75,75  | 0.98837209302326 |
| --val --st --net idsia_net.lua --cnn 150,200,300,350 --locnet 150,150,150 --locnet2 75,75,75  | 0.9937984496124 |
| --val --st --net idsia_net.lua --cnn 200,250,350,400 --locnet 150,150,150 --locnet2 75,75,75  | 0.99302325581395 |
| --val --st --net idsia_net.lua --cnn 150,200,300,350 --locnet 200,300,200 --locnet2 75,75,75  | 0.9937984496124 |
| --val --st --net idsia_net.lua --cnn 200,250,350,400 --locnet 200,300,200 --locnet2 75,75,75  | 0.9937984496124 |
| --val --st --net idsia_net.lua --cnn 150,200,300,350 --locnet ""  --locnet3 150,150,150 | 0.98914728682171 |
| --val --st --net idsia_net.lua --cnn 200,250,350,400 --locnet ""  --locnet3 150,150,150 | 0.9906976744186 |
| --val --st --net idsia_net.lua --cnn 150,200,300,350 --locnet 150,150,150  --locnet3 150,150,150 | 0.9937984496124 |
| --val --st --net idsia_net.lua --cnn 200,250,350,400 --locnet 150,150,150  --locnet3 150,150,150 | 0.9922480620155 |
| --val --st --net idsia_net.lua --cnn 150,200,300,350 --locnet 200,300,200  --locnet3 150,150,150 | 0.9953488372093 |
| --val --st --net idsia_net.lua --cnn 200,250,350,400 --locnet 200,300,200  --locnet3 150,150,150 | 0.99457364341085 |
| --val --st --net idsia_net.lua --cnn 150,200,300,350 --locnet "" --locnet2 150,150,150 --locnet3 150,150,150 | 0.99612403100775 |
| --val --st --net idsia_net.lua --cnn 200,250,350,400 --locnet "" --locnet2 150,150,150 --locnet3 150,150,150 | 0.99147286821705 |
| --val --st --net idsia_net.lua --cnn 150,200,300,350 --locnet 150,150,150 --locnet2 150,150,150 --locnet3 150,150,150 | 0.9953488372093 |
| --val --st --net idsia_net.lua --cnn 200,250,350,400 --locnet 150,150,150 --locnet2 150,150,150 --locnet3 150,150,150 | 0.99457364341085 |
| --val --st --net idsia_net.lua --cnn 150,200,300,350 --locnet 200,300,200 --locnet2 150,150,150 --locnet3 150,150,150 | 0.9953488372093 |
| --val --st --net idsia_net.lua --cnn 200,250,350,400 --locnet 200,300,200 --locnet2 150,150,150 --locnet3 150,150,150 | 0.9937984496124 |
| --val --st --net idsia_net.lua --cnn 150,200,300,350 --locnet "" --locnet2 75,75,75 --locnet3 150,150,150 | 0.98914728682171 |
| --val --st --net idsia_net.lua --cnn 200,250,350,400 --locnet "" --locnet2 75,75,75 --locnet3 150,150,150 | 0.99302325581395 |
| --val --st --net idsia_net.lua --cnn 150,200,300,350 --locnet 150,150,150 --locnet2 75,75,75 --locnet3 150,150,150 | 0.99612403100775 |
| --val --st --net idsia_net.lua --cnn 200,250,350,400 --locnet 150,150,150 --locnet2 75,75,75 --locnet3 150,150,150 | 0.9937984496124 |
| --val --st --net idsia_net.lua --cnn 150,200,300,350 --locnet 200,300,200 --locnet2 75,75,75 --locnet3 150,150,150 | 0.9953488372093 |
| --val --st --net idsia_net.lua --cnn 200,250,350,400 --locnet 200,300,200 --locnet2 75,75,75 --locnet3 150,150,150 | 0.9922480620155 |
| --val --st --net idsia_net.lua --cnn 150,200,300,350 --locnet ""  --locnet3 75,75,75 | 0.98837209302326 |
| --val --st --net idsia_net.lua --cnn 200,250,350,400 --locnet ""  --locnet3 75,75,75 | 0.98372093023256 |
| --val --st --net idsia_net.lua --cnn 150,200,300,350 --locnet 150,150,150  --locnet3 75,75,75 | 0.9937984496124 |
| --val --st --net idsia_net.lua --cnn 200,250,350,400 --locnet 150,150,150  --locnet3 75,75,75 | 0.9937984496124 |
| --val --st --net idsia_net.lua --cnn 150,200,300,350 --locnet 200,300,200  --locnet3 75,75,75 | 0.99612403100775 |
| --val --st --net idsia_net.lua --cnn 200,250,350,400 --locnet 200,300,200  --locnet3 75,75,75 | 0.99457364341085 |
| --val --st --net idsia_net.lua --cnn 150,200,300,350 --locnet "" --locnet2 150,150,150 --locnet3 75,75,75 | 0.99147286821705 |
| --val --st --net idsia_net.lua --cnn 200,250,350,400 --locnet "" --locnet2 150,150,150 --locnet3 75,75,75 | 0.9937984496124 |
| --val --st --net idsia_net.lua --cnn 150,200,300,350 --locnet 150,150,150 --locnet2 150,150,150 --locnet3 75,75,75 | 0.9953488372093 |
| --val --st --net idsia_net.lua --cnn 200,250,350,400 --locnet 150,150,150 --locnet2 150,150,150 --locnet3 75,75,75 | 0.9937984496124 |
| --val --st --net idsia_net.lua --cnn 150,200,300,350 --locnet 200,300,200 --locnet2 150,150,150 --locnet3 75,75,75 | 0.99302325581395 |
| --val --st --net idsia_net.lua --cnn 200,250,350,400 --locnet 200,300,200 --locnet2 150,150,150 --locnet3 75,75,75 | 0.9937984496124 |
