cmake_minimum_required(VERSION 3.16)
project(vision_tools LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio calib3d highgui)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)

if(TARGET yaml-cpp::yaml-cpp)
  set(VISION_TOOLS_YAML_TARGET yaml-cpp::yaml-cpp)
else()
  set(VISION_TOOLS_YAML_TARGET yaml-cpp)
endif()

add_library(vision_tools_recording STATIC
  src/recording/recorder.cpp
  src/calibration/timed_image_saver.cpp)
target_include_directories(vision_tools_recording PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(vision_tools_recording PUBLIC Eigen3::Eigen ${OpenCV_LIBS}
  ${VISION_TOOLS_YAML_TARGET})

add_library(vision_tools_calibration STATIC src/calibration/high_precision_calibrator.cpp)
target_include_directories(vision_tools_calibration PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(vision_tools_calibration PUBLIC ${OpenCV_LIBS}
  ${VISION_TOOLS_YAML_TARGET})

add_library(vision_tools_replay STATIC src/replay/frame_timeline.cpp)
target_include_directories(vision_tools_replay PUBLIC include)

add_library(vision_tools_web_ipc STATIC src/web/web_sink.cpp src/web/web_cmd.cpp)
target_include_directories(vision_tools_web_ipc PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(vision_tools_web_ipc PUBLIC ${OpenCV_LIBS})

add_executable(vision_tools_example_record examples/record_synthetic.cpp)
target_link_libraries(vision_tools_example_record PRIVATE vision_tools_recording)

add_executable(vision_tools_example_calibrate examples/calibrate.cpp)
target_link_libraries(vision_tools_example_calibrate PRIVATE vision_tools_calibration)

add_executable(vision_tools_example_replay_info examples/replay_info.cpp)
target_include_directories(vision_tools_example_replay_info PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(vision_tools_example_replay_info PRIVATE vision_tools_replay ${OpenCV_LIBS})

include(CTest)
if(BUILD_TESTING)
  foreach(tool IN ITEMS recorder timed_image_saver camera_calibrator frame_timeline)
    add_executable(vision_tools_${tool}_smoke tests/${tool}_smoke.cpp)
    if(tool STREQUAL "recorder" OR tool STREQUAL "timed_image_saver")
      target_link_libraries(vision_tools_${tool}_smoke PRIVATE vision_tools_recording)
    elseif(tool STREQUAL "camera_calibrator")
      target_link_libraries(vision_tools_${tool}_smoke PRIVATE vision_tools_calibration)
    else()
      target_link_libraries(vision_tools_${tool}_smoke PRIVATE vision_tools_replay)
    endif()
    add_test(NAME vision_tools_${tool}_smoke COMMAND vision_tools_${tool}_smoke)
  endforeach()
  add_executable(vision_tools_web_ipc_smoke tests/web_ipc_smoke.cpp)
  target_link_libraries(vision_tools_web_ipc_smoke PRIVATE vision_tools_web_ipc)
  add_test(NAME vision_tools_web_ipc_smoke COMMAND vision_tools_web_ipc_smoke)
endif()
