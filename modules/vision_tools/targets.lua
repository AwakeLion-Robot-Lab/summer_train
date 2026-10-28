-- Can be included from the main project or built with this module's xmake.lua.
local module_dir = os.scriptdir()
local function source(name) return path.join(module_dir, name) end
-- Match the host project's nlohmann/json ABI when this module is embedded.
-- A standalone copy simply uses the system header.
local host_json_include = path.join(module_dir, "../../tools/logger/include/3rdparty")

target("vision_tools_recording")
    set_kind("static")
    set_default(false)
    add_files(source("src/recording/recorder.cpp"),
              source("src/calibration/timed_image_saver.cpp"))
    add_includedirs(source("include"), "/usr/include/eigen3", "/usr/include/opencv4", {public = true})
    add_links("opencv_core", "opencv_imgcodecs", "opencv_videoio", "yaml-cpp", {public = true})
    add_syslinks("pthread", {public = true})
target_end()

target("vision_tools_calibration")
    set_kind("static")
    set_default(false)
    add_files(source("src/calibration/high_precision_calibrator.cpp"))
    add_includedirs(source("include"), "/usr/include/opencv4", {public = true})
    add_links("opencv_core", "opencv_imgproc", "opencv_imgcodecs",
              "opencv_calib3d", "opencv_highgui", "yaml-cpp", {public = true})
target_end()

target("vision_tools_replay")
    set_kind("static")
    set_default(false)
    add_files(source("src/replay/frame_timeline.cpp"))
    add_includedirs(source("include"), {public = true})
target_end()

target("vision_tools_web_ipc")
    set_kind("static")
    set_default(false)
    add_files(source("src/web/web_sink.cpp"), source("src/web/web_cmd.cpp"))
    add_includedirs(source("include"), "/usr/include/opencv4", {public = true})
    if os.isdir(host_json_include) then
        add_includedirs(host_json_include, {public = true})
    end
    add_links("opencv_core", "opencv_imgproc", "opencv_imgcodecs", {public = true})
    add_syslinks("pthread", "rt", {public = true})
target_end()

target("vision_tools_frame_timeline_smoke")
    set_kind("binary")
    set_default(false)
    add_files(source("tests/frame_timeline_smoke.cpp"))
    add_deps("vision_tools_replay")
target_end()

target("vision_tools_recorder_smoke")
    set_kind("binary")
    set_default(false)
    add_files(source("tests/recorder_smoke.cpp"))
    add_deps("vision_tools_recording")
target_end()

target("vision_tools_timed_image_saver_smoke")
    set_kind("binary")
    set_default(false)
    add_files(source("tests/timed_image_saver_smoke.cpp"))
    add_deps("vision_tools_recording")
target_end()

target("vision_tools_camera_calibrator_smoke")
    set_kind("binary")
    set_default(false)
    add_files(source("tests/camera_calibrator_smoke.cpp"))
    add_deps("vision_tools_calibration")
target_end()

target("vision_tools_web_ipc_smoke")
    set_kind("binary")
    set_default(false)
    add_files(source("tests/web_ipc_smoke.cpp"))
    add_deps("vision_tools_web_ipc")
target_end()

target("vision_tools_example_record")
    set_kind("binary")
    set_default(false)
    add_files(source("examples/record_synthetic.cpp"))
    add_deps("vision_tools_recording")
target_end()

target("vision_tools_example_calibrate")
    set_kind("binary")
    set_default(false)
    add_files(source("examples/calibrate.cpp"))
    add_deps("vision_tools_calibration")
target_end()

target("vision_tools_example_replay_info")
    set_kind("binary")
    set_default(false)
    add_files(source("examples/replay_info.cpp"))
    add_deps("vision_tools_replay")
    add_includedirs("/usr/include/opencv4")
    add_links("opencv_core", "opencv_videoio")
target_end()
