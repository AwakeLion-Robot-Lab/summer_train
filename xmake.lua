set_project("newvision")
set_version("0.1.0")
set_xmakever("2.9.8")
set_languages("c++20")
set_toolchains("gcc-13")

add_rules("mode.debug", "mode.release")

if is_plat("windows") then
    add_cxxflags("/utf-8")
end

option("use_xrepo_deps")
    set_default(false)
    set_showmenu(true)
    set_description("Use xmake-repo packages for OpenCV and yaml-cpp")
option_end()

option("use_system_deps")
    set_default(true)
    set_showmenu(true)
    set_description("Use system OpenCV and yaml-cpp development packages")
option_end()

if has_config("use_xrepo_deps") then
    add_requires("opencv", {optional = true})
    add_requires("yaml-cpp", {optional = true})
end

target("newvision")
    set_kind("static")
    add_files("src/**/*.cpp")
    add_files("tools/camera_sdk/hikrobot/hikrobot.cpp")
    add_files("tools/camera_sdk/mindvision/mindvision.cpp")
    add_headerfiles("include/**/*.hpp")
    add_includedirs("include", {public = true})
    add_includedirs("tools/config_set/include", {public = true})
    add_includedirs("tools/LatesBuffer/include", {public = true})
    add_includedirs("tools/camera_sdk", {public = true})
    add_includedirs("tools/camera_sdk/hikrobot", {public = true})
    add_includedirs("tools/camera_sdk/hikrobot/include", {public = true})
    add_includedirs("tools/camera_sdk/mindvision", {public = true})
    add_includedirs("tools/camera_sdk/mindvision/include", {public = true})
    add_includedirs("tools/logger/include", {public = true})
    add_includedirs("tools/logger/include/3rdparty", {public = true})
    add_linkdirs("tools/camera_sdk/hikrobot/lib/amd64", "tools/camera_sdk/mindvision/lib/amd64", {public = true})
    add_rpathdirs("$(projectdir)/tools/camera_sdk/hikrobot/lib/amd64", "$(projectdir)/tools/camera_sdk/mindvision/lib/amd64", {public = true})
    add_links("MvCameraControl", "MVSDK", "usb-1.0", {public = true})
    if has_config("use_xrepo_deps") then
        add_packages("opencv", "yaml-cpp", {public = true})
    elseif has_config("use_system_deps") then
        add_includedirs("/usr/include/opencv4", {public = true})
        add_links("opencv_core", "opencv_imgproc", "opencv_imgcodecs", "opencv_videoio", "opencv_calib3d", "opencv_dnn", "opencv_highgui", "yaml-cpp", {public = true})
    end

target("auto_aim")
    set_kind("binary")
    set_default(false)
    set_rundir("$(projectdir)")
    add_files("tests/main.cpp")
    add_deps("newvision")

target("logger_smoke")
    set_kind("binary")
    set_default(false)
    set_rundir("$(projectdir)")
    add_files("tests/logger_smoke.cpp")
    add_files("src/l6_telemetry/logger.cpp")
    add_includedirs("include")
    add_includedirs("tools/logger/include")
    add_includedirs("tools/logger/include/3rdparty")

target("latest_buffer_smoke")
    set_kind("binary")
    set_default(false)
    set_rundir("$(projectdir)")
    add_files("tests/latest_buffer_smoke.cpp")
    add_includedirs("tools/LatesBuffer/include")

target("fps_counter_smoke")
    set_kind("binary")
    set_default(false)
    set_rundir("$(projectdir)")
    add_files("tests/fps_counter_smoke.cpp")
    add_files("src/l6_telemetry/fps_counter.cpp")
    add_includedirs("include")
