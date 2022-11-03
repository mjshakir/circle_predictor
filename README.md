# Circle Predictor
This use the ```libtorch``` to create a simple circle predictor just as an example.**This is not a perfect network, but feel free to imptove it.**.
In this example, the network is fed an **X** value/point from the cartesian coordinate system. The results is the **Y** point and its opposite point in the circle.  

# How to build
- Clone the repo: 
```
git clone https://github.com/mjshakir/circle_predictor.git
```
- Pull the submodules: 
```
git pull --recurse-submodules
```
- A shortcut is to clone with submodules
```
git clone https://github.com/mjshakir/circle_predictor.git --recurse-submodules
```

## Get libtorch
- Go to [PyTorch](https://pytorch.org/) to get ```libtorch```
- Unziping 
```
sudo apt-get install -y unzip 
```
```
unzip path/to/libtorch.zip
```
- Copy ```libtorch``` to the project folder 
```
cp -r ~/path/to/libtorch ~/path/to/external_library/circle_predictor
```
- Folder system should look like 
```
circle_predictor/
├── CMakeLists.txt
├── external_library
    └── libtorch
├── include
├── main
└── src
``` 
## Build
- Install ```boost``` development and ```TTB``` development:
```
sudo apt-get install -y libboost-all-dev libtbb-dev
```
- Get Ninja: 
```
sudo apt-get install -y ninja-build
```
### Build Ninja
- Enter the project path
```
cd ~/path/to/circle_predictor/
```
- Build Ninja: 
```
cmake -DCMAKE_BUILD_TYPE=release -B build -G Ninja
```
  - Build Ninja using ```CLang```:
```
cmake -DCMAKE_CXX_COMPILER=/usr/bin/clang++ -DCMAKE_BUILD_TYPE=release -B build  -G Ninja
```  
- Build the project: 
```
cd build/; ninja
```
### TTB issue
- Build Ninja: 
```
cmake -DUSE_TTB=OFF -DCMAKE_BUILD_TYPE=release -B build  -G Ninja
```
  - Build Ninja using ```CLang```:
```
cmake -DUSE_TTB=OFF -DCMAKE_CXX_COMPILER=/usr/bin/clang++ -DCMAKE_BUILD_TYPE=release -B build  -G Ninja
```  
- Build the project: 
```
cd build/; ninja
```
# Run the program
- Run: (*Reminder: be in ```~/path/to/circle_predictor/build/```*) 
```
./bin/circle_predictor
```
## Command line arguments
- ```-h``` or ```--help```: 
  - Display options and description:
- ```-s``` or ```--filename```: default ```test_results```
  - Save the results of the test data in ```csv``` file.
- ```-t``` or ```---training_size```: default ```100```
  - How many different points to train
  - Accepts an integer ```x > 0```
- ```-g``` or ```--generated_size```: default ```10000```
  - How many points generated
  - Accepts an integer ```x >= 100```
- ```-b``` or ```--batch_size```: default ```20```
  - Batch the generated data to train 
  - Limitations:
    - Accept an unsigned integer ```x > 0```
    - Must be less then the ```generated_size```
    - Must be less the ```1000``` this a ```libtorch``` limitation
- ```-u``` or ```--use_epoch```: default ```false```
  - Train with an epoch iterations or validation precision
- ```-p``` or ```--precision```: default ```2.5E-1L``` 
  - Determine when to stop the training. This uses a validation set.
- ```-e``` or ```--epoch```: default ```20```
  - How many iteration to train.

### Example: 
- With precision and validation:
``` 
./bin/circle_predictor precision_results 10 1200 false 1E-2
```
- With Epoch iteration:
``` 
./bin/circle_predictor epoch_results 10 1200 true 10
```
