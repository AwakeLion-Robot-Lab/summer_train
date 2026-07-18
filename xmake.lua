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

option("use_openvino")
    set_default(false)
    set_showmenu(true)
    set_description("Enable the optional OpenVINO inference backend")
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
    add_files("tools/serial/src/serial.cc")
    add_files("tools/serial/src/impl/unix.cc")
    add_headerfiles("include/**/*.hpp")
    add_headerfiles("tools/serial/include/**/*.h")
    add_includedirs("include", {public = true})
    add_includedirs("/usr/include/eigen3", {public = true})
    add_includedirs("tools/serial/include", {public = true})
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
    add_syslinks("pthread", "rt", {public = true})
    if has_config("use_xrepo_deps") then
        add_packages("opencv", "yaml-cpp", {public = true})
    elseif has_config("use_system_deps") then
        add_includedirs("/usr/include/opencv4", {public = true})
        add_links("opencv_core", "opencv_imgproc", "opencv_imgcodecs", "opencv_videoio", "opencv_calib3d", "opencv_dnn", "opencv_highgui", "yaml-cpp", {public = true})
    end
    if has_config("use_openvino") then
        -- 只使用已经安装并由 pkg-config 暴露的 OpenVINO，避免 xrepo 再下载一套 SDK。
        on_load(function (target)
            import("lib.detect.find_package")
            local openvino = find_package("pkgconfig::openvino", {version = true})
            assert(openvino,
                "OpenVINO was not found through pkg-config; source setupvars.sh first")

            target:add("includedirs", openvino.includedirs)
            if openvino.defines then
                target:add("defines", openvino.defines)
            end
            target:add("defines", "NEWVISION_HAS_OPENVINO=1")

            -- newvision 是静态库，最终可执行文件仍需要继承 Runtime 链接和 RUNPATH。
            target:add("linkdirs", openvino.linkdirs, {public = true})
            target:add("rpathdirs", openvino.linkdirs, {public = true})
            target:add("links", "openvino", {public = true})
        end)
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

target("camera_calibration_smoke")
    set_kind("binary")
    set_default(false)
    set_rundir("$(projectdir)")
    add_files("tests/camera_calibration_smoke.cpp")
    add_deps("newvision")

target("pnp_solver_smoke")
    set_kind("binary")
    set_default(false)
    set_rundir("$(projectdir)")
    add_files("tests/pnp_solver_smoke.cpp")
    add_deps("newvision")

target("auto_aim_types_smoke")
    set_kind("binary")
    set_default(false)
    set_rundir("$(projectdir)")
    add_files("tests/auto_aim_types_smoke.cpp")
    add_deps("newvision")

target("fps_counter_smoke")
    set_kind("binary")
    set_default(false)
    set_rundir("$(projectdir)")
    add_files("tests/fps_counter_smoke.cpp")
    add_files("src/l6_telemetry/fps_counter.cpp")
    add_includedirs("include")

target("udp_json_sender_smoke")
    set_kind("binary")
    set_default(false)
    set_rundir("$(projectdir)")
    add_files("tests/udp_json_sender_smoke.cpp")
    add_files("src/l6_telemetry/udp_json_sender.cpp")
    add_includedirs("include")
    add_includedirs("tools/logger/include/3rdparty")
    add_syslinks("pthread")

if has_config("use_openvino") then
target("openvino_armor_smoke")
    set_kind("binary")
    set_default(false)
    set_rundir("$(projectdir)")
    add_files("tests/openvino_armor_smoke.cpp")
    add_deps("newvision")
end

target("serial_protocol_smoke")
    set_kind("binary")
    set_default(false)
    set_rundir("$(projectdir)")
    add_files("tests/serial_protocol_smoke.cpp")
    add_files("src/l1_sensor/serial/serial_protocol.cpp")
    add_files("src/l6_telemetry/logger.cpp")
    add_includedirs("include")
    add_includedirs("tools/logger/include")
    add_includedirs("tools/logger/include/3rdparty")

if is_plat("linux") then
target("serial_worker_smoke")
    set_kind("binary")
    set_default(false)
    set_rundir("$(projectdir)")
    add_files("tests/serial_worker_smoke.cpp")
    add_deps("newvision")
    add_syslinks("util")
end

target("serial_hardware_smoke")
    set_kind("binary")
    set_default(false)
    set_rundir("$(projectdir)")
    add_files("tests/serial_hardware_smoke.cpp")
    add_deps("newvision")
