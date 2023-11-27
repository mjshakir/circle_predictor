cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(circle_predictor)
#------------------------------------------------------------------------------------------
# Default to C++20
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()
#------------------------------------------------------------------------------------------
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
#------------------------------------------------------------------------------------------
option (FORCE_COLORED_OUTPUT "Always produce ANSI-colored output (GNU/Clang only)." OFF)
MESSAGE(STATUS "FORCE_COLORED_OUTPUT: " ${FORCE_COLORED_OUTPUT})
#---------------
if (${FORCE_COLORED_OUTPUT})
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
       add_compile_options (-fdiagnostics-color=always)
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
       add_compile_options (-fcolor-diagnostics)
    endif ()
endif ()
#------------------------------------------------------------------------------------------
function(set_with_default variable help_text type value)
  set("${variable}" "<DEFAULT>" CACHE "${type}" "${help_text}")
  if (${variable} STREQUAL "<DEFAULT>")
    set("${variable}" "${value}" PARENT_SCOPE)
  endif ()
endfunction()
#------------------------------------------------------------------------------------------
# option(USE_CLANG "build application with CLang" OFF) # OFF is the default
# MESSAGE(STATUS "USE_CLANG: " ${USE_CLANG})
#---------------
# if(${USE_CLANG})
#   set(CMAKE_CXX_COMPILER "/usr/bin/clang++")
# endif(${USE_CLANG})
#------------------------------------------------------------------------------------------
# option(USE_TTB "build application with TTB" ON) # ON is the default
# MESSAGE(STATUS "USE_TTB: " ${USE_TTB})
#------------------------------------------------------------------------------------------
set_with_default(USE_TTB "build application with TTB" BOOL ON)
#------------------------------------------------------------------------------------------
add_library(common_options INTERFACE)
target_compile_options(common_options INTERFACE -Wall -Wextra -Wpedantic -pedantic-errors -Werror=format-security -fasynchronous-unwind-tables)
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
# set(BUILD_THREADPOOL_SHARED_LIBS ON) # for static library
add_subdirectory(${CMAKE_SOURCE_DIR}/external_library/ThreadPool)
#------------------------------------------------------------------------------------------
# set(BUILD_PROGRESSBAR_SHARED_LIBS ON) # for static library
add_subdirectory(${CMAKE_SOURCE_DIR}/external_library/ProgressBar)
#------------------------------------------------------------------------------------------
# set(BUILD_TIMING_SHARED_LIBS  ON) # for static library
add_subdirectory(${CMAKE_SOURCE_DIR}/external_library/TimeIt)
#------------------------------------------------------------------------------------------
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
#------------------------------------------------------------------------------------------
# Try to find fmt
find_package(fmt QUIET)
#---------------
if(fmt_FOUND)
    message(STATUS "Found fmt library")
    # set(HAVE_FMT TRUE)
    # add_definitions(-DHAVE_FMT)
else()
    message(WARNING "fmt library not found. Some features might not work.")
    # set(HAVE_FMT FALSE)
endif()
#---------------
find_package(Torch REQUIRED)
#---------------
if (NOT Torch_FOUND)
  message(FATAL_ERROR "Fatal error: Torch required.")
else()
  message(STATUS "Setting up Torch")
  message(STATUS "Includes - ${TORCH_INCLUDE_DIRS}")
  message(STATUS "Library  - ${TORCH_LIBRARIES}")
endif (NOT Torch_FOUND)
#----------------------------------
if(USE_TTB)
  #---------------
  find_package(TBB REQUIRED)
    #---------------
    if (NOT TBB_FOUND)
      message(FATAL_ERROR "Fatal error: TBB required.")
    else()
      message(STATUS "Setting up TBB")
      message(STATUS "Includes - ${TBB_INCLUDE_DIRS}")
      message(STATUS "Library  - ${TBB_LIBRARIES}")
    endif (NOT TBB_FOUND)
    #---------------
endif(USE_TTB)
#----------------------------------
find_package(Boost COMPONENTS program_options REQUIRED)
#----------------------------------
LINK_DIRECTORIES( ${LIBRARYPATH} )
#------------------------------------------------------------------------------------------
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
#------------------------------------------------------------------------------------------
add_library(${PROJECT_NAME}_network SHARED
  src/Network/Networks.cpp
  src/Network/Supervised/Supervised.cpp
  src/Network/Supervised/NetworkHandling.cpp
  src/Network/Supervised/DataLoader.cpp
)
#------------------------------------------------------------------------------------------
add_library(${PROJECT_NAME}_data SHARED
  src/Generate/Supervised/Generate.cpp
  src/Generate/Supervised/Normalize.cpp
)
#------------------------------------------------------------------------------------------
# add_library(${PROJECT_NAME}_timing SHARED
#   src/Timing/Timing.cpp
#   src/Timing/TimeIT.cpp
# )
#------------------------------------------------------------------------------------------
# RL Library
#----------------------------------
add_library(${PROJECT_NAME}_rl_data SHARED
  src/Generate/RL/Generate.cpp  
  src/Generate/RL/GeneratePoints.cpp
  src/Generate/RL/RLNormalize.cpp
  src/Generate/Supervised/Normalize.cpp
)
#------------------------------------------------------------------------------------------
add_library(${PROJECT_NAME}_rl_environment SHARED
  src/Environment/RL/Environment.cpp
  src/Environment/RL/RLEnvironment.cpp
  src/Environment/RL/RLEnvironmentLoader.cpp
  src/Environment/RL/RLEnvironmentShuffle.cpp
  src/Environment/RL/RLEnvironmentShuffleLoader.cpp
  src/Environment/RL/RLEnvironmentAtomic.cpp
  src/Environment/RL/RLEnvironmentLoaderAtomic.cpp
  src/Environment/RL/RLEnvironmentShuffleAtomic.cpp
  src/Environment/RL/RLEnvironmentShuffleAtomicLoader.cpp
  src/Environment/RL/RLEnvironmentTest.cpp
  src/Environment/RL/RLEnvironmentTestLoader.cpp
)
#------------------------------------------------------------------------------------------
add_library(${PROJECT_NAME}_rl_network SHARED
  src/Network/Networks.cpp
  src/Network/RL/ReinforcementNetworkHandling.cpp
  src/Network/RL/ReinforcementNetworkHandlingDQN.cpp
  src/Network/RL/ExperienceReplay.cpp
  src/Network/RL/ExperienceReplayBuffer.cpp
  src/Network/RL/Train.cpp
)
#------------------------------------------------------------------------------------------
# Miscellaneous Library
#----------------------------------
add_library(${PROJECT_NAME}_utilities SHARED
  src/Utilities/CircleEquation.cpp
  src/Utilities/StaticCheck.cpp
)
#------------------------------------------------------------------------------------------
if(USE_TTB)
  target_link_libraries(${PROJECT_NAME}_network ${TIMEIT_LIBRARIES} ${TORCH_LIBRARIES} ${PROGRESSBAR_LIBRARIES} ${THREADPOOL_LIBRARIES} tbb fort)
  target_link_libraries(${PROJECT_NAME}_data PRIVATE ${TORCH_LIBRARIES} tbb)
  target_link_libraries(${PROJECT_NAME}_rl_data PRIVATE ${TORCH_LIBRARIES} tbb)
  target_link_libraries(${PROJECT_NAME}_rl_environment PRIVATE ${TORCH_LIBRARIES} tbb)
  target_link_libraries(${PROJECT_NAME}_rl_network ${TORCH_LIBRARIES} ${THREADPOOL_LIBRARIES} ${PROGRESSBAR_LIBRARIES} tbb fort )
  target_link_libraries(${PROJECT_NAME}_utilities PRIVATE ${TORCH_LIBRARIES} tbb)
else()
  target_link_libraries(${PROJECT_NAME}_network ${TIMEIT_LIBRARIES} ${TORCH_LIBRARIES} ${PROGRESSBAR_LIBRARIES} ${THREADPOOL_LIBRARIES} fort)
  target_link_libraries(${PROJECT_NAME}_data PRIVATE ${TORCH_LIBRARIES})
  target_link_libraries(${PROJECT_NAME}_rl_data PRIVATE ${TORCH_LIBRARIES})
  target_link_libraries(${PROJECT_NAME}_rl_environment PRIVATE ${TORCH_LIBRARIES})
  target_link_libraries(${PROJECT_NAME}_rl_network ${TORCH_LIBRARIES} ${THREADPOOL_LIBRARIES} ${PROGRESSBAR_LIBRARIES} fort)
  target_link_libraries(${PROJECT_NAME}_utilities PUBLIC ${TORCH_LIBRARIES})
endif(USE_TTB)
#------------------------------------------------------------------------------------------
include_directories(
  include
  external_library
)
#------------------------------------------------------------------------------------------
add_executable(${PROJECT_NAME} main/main.cpp)
add_executable(${PROJECT_NAME}_rl main/rl_main.cpp)
#------------------------------------------------------------------------------------------
target_link_libraries(${PROJECT_NAME} PRIVATE
  ${PROJECT_NAME}_network 
  ${PROJECT_NAME}_data 
  ${TORCH_LIBRARIES}
  ${Boost_LIBRARIES}
  ${THREADPOOL_LIBRARIES}
  ${TIMEIT_LIBRARIES}
  common_options
  )
#------------------------------------------------------------------------------------------
target_link_libraries(${PROJECT_NAME}_rl PRIVATE
  ${PROJECT_NAME}_rl_data
  ${PROJECT_NAME}_rl_environment
  ${PROJECT_NAME}_rl_network
  ${PROJECT_NAME}_utilities
  ${TORCH_LIBRARIES}
  ${Boost_LIBRARIES}
  ${THREADPOOL_LIBRARIES}
  ${TIMEIT_LIBRARIES}
  fort
  common_options
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