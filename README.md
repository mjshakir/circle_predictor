# Circle Predictor
This use the ```libtorch``` to create a simple circle predictor just as an example.**This is not a perfect network, but feel free to imptove it.**.
In this example, the network is fed an **X** value/point from the cartesian coordinate system. The results is the **Y** point and its opposite point in the circle.  

# How to build
- Clone the repo: 
```
git clone https://github.com/Majedshakir/circle_predictor.git
```
- Pull the submodules: 
```
git pull --recurse-submodules
```
- A shortcut is to clone with submodules
```
git clone https://github.com/Majedshakir/circle_predictor.git --recurse-submodules
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
* Create a build folder 
```
~/path/to/circle_predictor/; mkdir build; cd build
```
* Get Ninja: 
```
sudo apt-get install -y ninja-build
```
* Build Ninja: 
```
cmake -DCMAKE_BUILD_TYPE=release .. -G Ninja
```
  * Build Ninja using ```CLang```:
```
cmake -DUSE_CLANG=ON -DCMAKE_BUILD_TYPE=release .. -G Ninja
```  
* Build the project: 
```
ninja
```

# Run the program
- Run: (*Reminder: be in ```~/path/to/circle_predictor/build/```*) 
```
./bin/circle_predictor
```
