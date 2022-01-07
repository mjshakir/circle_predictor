cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(circle_predictor)
#------------------------------------------------------------------------------------------
# Default to C++20
# if(NOT CMAKE_CXX_STANDARD)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_COMPILER /usr/bin/clang++)
# endif()
add_compile_options(-Wall -Wextra -Wpedantic)
#------------------------------------------------------------------------------------------
set(CMAKE_PREFIX_PATH ~/Documents/Projects/circle_predictor/libtorch)
#------------------------------------------------------------------------------------------
include_directories(${CMAKE_SOURCE_DIR}/libtorch/include)
link_directories(${CMAKE_SOURCE_DIR}/libtorch/lib)
#------------------------------------------------------------------------------------------
# add_library(Torch SHARED) # or STATIC instead of SHARED
#------------------------------------------------------------------------------------------
find_package(Torch REQUIRED)
find_package(OpenMP)
# find_package(TBB REQUIRED)
LINK_DIRECTORIES( ${LIBRARYPATH} )
#------------------------------------------------------------------------------------------
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#------------------------------------------------------------------------------------------
add_library(${PROJECT_NAME}_library SHARED
  src/Generate/Generate.cpp
  src/Network/NetworkHandling.cpp
  src/Network/DataLoader.cpp
  src/Timing/Timing.cpp
  src/Timing/TimingFunction.cpp
)
#------------------------------------------------------------------------------------------
add_dependencies(${PROJECT_NAME}_library "${TORCH_LIBRARIES}")
target_link_libraries(${PROJECT_NAME}_library "${TORCH_LIBRARIES}")
#------------------------------------------------------------------------------------------
include_directories(
  include
)
#------------------------------------------------------------------------------------------
add_executable(${PROJECT_NAME} Main/main.cpp)
#------------------------------------------------------------------------------------------
target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_library "${TORCH_LIBRARIES}")
#------------------------------------------------------------------------------------------
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)
#------------------------------------------------------------------------------------------
#=============
# Install
#=============
install(
  DIRECTORY include/
  DESTINATION include
)
#------------------------------------------------------------------------------------------
install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)
# #------------------------------------------------------------------------------------------