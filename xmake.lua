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
        -- SP-Vision 固定使用 /opt/intel/openvino_2024.6.0。模型推理的最后几个
        -- ulp 会随 Runtime 版本变化，而 SP 的 1 度离散 yaw 搜索会放大这种差异，
        -- 所以本机存在同一 SDK 时优先与 SP 链接同一版本；其他机器再回退 pkg-config。
        on_load(function (target)
            local sp_openvino_runtime = "/opt/intel/openvino_2024.6.0/runtime"
            local sp_openvino_include = path.join(sp_openvino_runtime, "include")
            local sp_openvino_lib = path.join(sp_openvino_runtime, "lib", "intel64")
            if os.isdir(sp_openvino_include) and os.isfile(path.join(sp_openvino_lib, "libopenvino.so")) then
                target:add("includedirs", sp_openvino_include)
                target:add("linkdirs", sp_openvino_lib, {public = true})
                target:add("rpathdirs", sp_openvino_lib, {public = true})
            else
                import("lib.detect.find_package")
                local openvino = find_package("pkgconfig::openvino", {version = true})
                assert(openvino,
                    "OpenVINO was not found; install 2024.6 like SP or expose it through pkg-config")
                target:add("includedirs", openvino.includedirs)
                if openvino.defines then
                    target:add("defines", openvino.defines)
                end
                target:add("linkdirs", openvino.linkdirs, {public = true})
                target:add("rpathdirs", openvino.linkdirs, {public = true})
            end
            target:add("defines", "NEWVISION_HAS_OPENVINO=1")
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

target("armor_decoder_smoke")
    set_kind("binary")
    set_default(false)
    set_rundir("$(projectdir)")
    add_files("tests/armor_decoder_smoke.cpp")
    add_deps("newvision")

target("image_preprocessor_smoke")
    set_kind("binary")
    set_default(false)
    set_rundir("$(projectdir)")
    add_files("tests/image_preprocessor_smoke.cpp")
    add_deps("newvision")

-- 灯条精修只依赖 OpenCV，不牵扯相机/串口 SDK，因此直接列出所需文件而不是
-- add_deps("newvision")，保证在没有硬件库的机器上也能单独构建。
target("armor_refiner_smoke")
    set_kind("binary")
    set_default(false)
    set_rundir("$(projectdir)")
    add_files("tests/armor_refiner_smoke.cpp")
    add_files("src/l2_perception/armor/armor_refiner.cpp")
    add_includedirs("include")
    add_includedirs("/usr/include/eigen3")
    if has_config("use_xrepo_deps") then
        add_packages("opencv")
    elseif has_config("use_system_deps") then
        add_includedirs("/usr/include/opencv4")
        add_links("opencv_core", "opencv_imgproc")
    end

target("planner_smoke")
    set_kind("binary")
    set_default(false)
    set_rundir("$(projectdir)")
    add_files("tests/planner_smoke.cpp")
    add_deps("newvision")

target("fire_decision_smoke")
    set_kind("binary")
    set_default(false)
    set_rundir("$(projectdir)")
    add_files("tests/fire_decision_smoke.cpp")
    add_deps("newvision")

target("tracker_smoke")
    set_kind("binary")
    set_default(false)
    set_rundir("$(projectdir)")
    add_files("tests/tracker_smoke.cpp")
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

target("auto_aim_test")
    set_kind("binary")
    set_default(false)
    set_rundir("$(projectdir)")
    add_files("tests/auto_aim_test.cpp")
    add_deps("newvision")

-- 整车跟踪链路的离线诊断，输出 CSV，不需要显示器。
target("track_diag")
    set_kind("binary")
    set_default(false)
    set_rundir("$(projectdir)")
    add_files("tests/track_diag.cpp")
    add_deps("newvision")

-- 灯条精修/筛选的离线回放，需要模型和显示器，因此和 auto_aim_test 一样只在
-- use_openvino=y 时存在。无显示器的机器请改跑 armor_refiner_smoke。
target("armor_refiner_video_test")
    set_kind("binary")
    set_default(false)
    set_rundir("$(projectdir)")
    add_files("tests/armor_refiner_video_test.cpp")
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
