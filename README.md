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
unzip path/to/libtorch.zip
```
- Copy ```libtorch``` to the project folder ```cp -r ~/path/to/libtorch ~/path/to/external_library/circle_predictor```
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
- Create a build folder 
```
~/path/to/circle_predictor/; mkdir build; cd build
```
- Get Ninja: 
```
sudo apt-get install -y ninja-build
```
- Build Ninja: 
```
cmake -DCMAKE_BUILD_TYPE=release .. -G Ninja
```
  - Build Ninja using ```CLang```:
```
cmake -DUSE_CLANG=ON -DCMAKE_BUILD_TYPE=release .. -G Ninja
```  
- Build the project: 
```
ninja
```

# Run the program
- Run: (*Reminder: be in ```~/path/to/circle_predictor/build/```*) 
```
./bin/circle_predictor
```
## Command line arguments
- ```filename```: default ```test_results```
  - Save the results of the test data in ```csv``` file.
- ```training_size```: default ```100```
  - How many different points to train
  - Accept an integer ```x > 0```
- ```generated_size```: default ```10000```
  - How many points generated
  - Accept an integer ```x >= 200```
- ```batch_size```: default ```20```
  - Batch the generated data to train 
  - Limitations:
    - Accept an unsigned integer ```x > 0```
    - Must be less then the ```generated_size```
    - Must be less the ```1000``` this a ```libtorch``` limitation
- ```isEpoch```: default ```false```
  - Train with an epoch iteration or validation precision
  - Accept a bool ```true``` or ```1```
- ```precision``` if ```isEpoch``` is ```false``` or ```epoch``` if ```isEpoch``` is ```true```
  - ```precision```: a long double that determine when to stop the training. This uses a validation set.
  - ```epoch```: how many iteration to train

### Example: 
- With precision and validation:
``` 
./bin/circle_predictor precision_results 10 1200 false 1E-2
```
- With Epoch iteration:
``` 
./bin/circle_predictor epoch_results 10 1200 true 10
```
