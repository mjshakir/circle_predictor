cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(circle_predictor)
#------------------------------------------------------------------------------------------
# Default to C++20
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()
#------------------------------------------------------------------------------------------
option(USE_CLANG "build application with CLang" OFF) # OFF is the default
SET(USE_CLANG OFF CACHE BOOL "Set CLang Option")
MESSAGE("-- USE_CLANG: " ${USE_CLANG})
#---------------
option(USE_TTB "build application with TTB" ON) # ON is the default
SET(USE_TTB ON CACHE BOOL "Set TTB Option")
MESSAGE("-- USE_TTB: " ${USE_TTB})
#---------------
#------------------------------------------------------------------------------------------
if(USE_CLANG EQUAL ON)
  set(CMAKE_CXX_COMPILER /usr/bin/clang++)
endif(USE_CLANG EQUAL ON)
#------------------------------------------------------------------------------------------
add_compile_options(-Wall -Wextra -Wpedantic)
#------------------------------------------------------------------------------------------
# Set the output folder where your program will be created
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
#------------------------------------------------------------------------------------------
set(CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/external_library/libtorch)
#------------------------------------------------------------------------------------------
include_directories(${CMAKE_SOURCE_DIR}/external_library/libtorch/include)
link_directories(${CMAKE_SOURCE_DIR}/external_library/libtorch/lib)
#------------------------------------------------------------------------------------------
# Disable building tests and examples in libfort project
set(FORT_ENABLE_TESTING OFF CACHE INTERNAL "")

add_subdirectory(${CMAKE_SOURCE_DIR}/external_library/libfort)
#------------------------------------------------------------------------------------------
find_package(Torch REQUIRED)
#---------------
if (NOT Torch_FOUND)
  message(FATAL_ERROR "Fatal error: Torch required.")
else()
  message(STATUS "-- Setting up Torch")
  message(STATUS "--- Includes - ${TORCH_INCLUDE_DIRS}")
  message(STATUS "--- Library  - ${TORCH_LIBRARIES}")
endif (NOT Torch_FOUND)
#----------------------------------
if(USE_TTB EQUAL ON)
  #---------------
  find_package(TBB REQUIRED)
    #---------------
    if (NOT TBB_FOUND)
      message(FATAL_ERROR "Fatal error: TBB required.")
    else()
      message(STATUS "-- Setting up TBB")
      message(STATUS "--- Includes - ${TBB_INCLUDE_DIRS}")
      message(STATUS "--- Library  - ${TBB_LIBRARIES}")
    endif (NOT TBB_FOUND)
    #---------------
endif(USE_TTB EQUAL ON)
#----------------------------------
LINK_DIRECTORIES( ${LIBRARYPATH} )
#------------------------------------------------------------------------------------------
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
#------------------------------------------------------------------------------------------
add_library(${PROJECT_NAME}_network SHARED
  src/Network/Network.cpp
  src/Network/Networks.cpp
  src/Network/NetworkHandling.cpp
  src/Network/DataLoader.cpp
  src/Network/Normalize.cpp
)
#------------------------------------------------------------------------------------------
add_library(${PROJECT_NAME}_data SHARED
  src/Generate/Generate.cpp
)
#------------------------------------------------------------------------------------------
add_library(${PROJECT_NAME}_timing SHARED
  src/Timing/Timing.cpp
  src/Timing/TimeIT.cpp
)
#------------------------------------------------------------------------------------------
if(USE_TTB EQUAL ON)
  target_link_libraries(${PROJECT_NAME}_network ${TORCH_LIBRARIES} tbb fort)
  target_link_libraries(${PROJECT_NAME}_data PRIVATE ${TORCH_LIBRARIES} tbb)
else()
  target_link_libraries(${PROJECT_NAME}_network ${TORCH_LIBRARIES} fort)
  target_link_libraries(${PROJECT_NAME}_data PRIVATE ${TORCH_LIBRARIES})
endif(USE_TTB EQUAL ON)
#------------------------------------------------------------------------------------------
include_directories(
  include
  external_library
)
#------------------------------------------------------------------------------------------
add_executable(${PROJECT_NAME} main/main.cpp)
#------------------------------------------------------------------------------------------
target_link_libraries(${PROJECT_NAME} PRIVATE
  ${PROJECT_NAME}_network 
  ${PROJECT_NAME}_data 
  ${PROJECT_NAME}_timing 
  ${TORCH_LIBRARIES}
  )
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