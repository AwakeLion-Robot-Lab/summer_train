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

option("use_tensorrt")
    set_default(false)
    set_showmenu(true)
    set_description("Enable the optional TensorRT/CUDA inference backend")
option_end()

option("tensorrt_root")
    set_default("")
    set_showmenu(true)
    set_description("TensorRT SDK prefix (used only when use_tensorrt=y)")
option_end()

option("cuda_root")
    set_default("")
    set_showmenu(true)
    set_description("CUDA SDK prefix (used only when use_tensorrt=y)")
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
    add_includedirs("tools/LatestBuffer/include", {public = true})
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
    -- OpenVINO 与 TensorRT 共用一个 on_load；xmake 对同一 target 的多个 on_load
    -- 使用覆盖语义，拆开写会让其中一个后端静默失效。
    on_load(function (target)
        if has_config("use_openvino") then
            -- SP-Vision 固定使用 /opt/intel/openvino_2024.6.0。模型推理的最后几个
            -- ulp 会随 Runtime 版本变化，而 SP 的 1 度离散 yaw 搜索会放大这种差异，
            -- 所以本机存在同一 SDK 时优先与 SP 链接同一版本；其他机器再回退 pkg-config。
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
        end

        if has_config("use_tensorrt") then
            local function append_unique(values, value)
                if value == nil or value == "" then
                    return
                end
                for _, existing in ipairs(values) do
                    if existing == value then
                        return
                    end
                end
                table.insert(values, value)
            end

            local tensorrt_roots = {}
            append_unique(tensorrt_roots, get_config("tensorrt_root"))
            append_unique(tensorrt_roots, os.getenv("TENSORRT_ROOT"))
            append_unique(tensorrt_roots, "/usr/local/TensorRT")
            append_unique(tensorrt_roots, "/usr/local")
            append_unique(tensorrt_roots, "/usr")

            local cuda_roots = {}
            append_unique(cuda_roots, get_config("cuda_root"))
            append_unique(cuda_roots, os.getenv("CUDA_HOME"))
            append_unique(cuda_roots, os.getenv("CUDA_PATH"))
            append_unique(cuda_roots, "/usr/local/cuda")
            append_unique(cuda_roots, "/usr")

            local function find_header(roots, names)
                for _, root in ipairs(roots) do
                    for _, name in ipairs(names) do
                        local candidate = path.join(root, name)
                        if os.isfile(candidate) then
                            return path.directory(candidate)
                        end
                    end
                end
                return nil
            end

            local function find_library_dir(roots, required_names)
                for _, root in ipairs(roots) do
                    for _, relative in ipairs({"lib", "lib64", "lib/x86_64-linux-gnu", "lib/aarch64-linux-gnu"}) do
                        local directory = path.join(root, relative)
                        local found = true
                        for _, name in ipairs(required_names) do
                            if not os.isfile(path.join(directory, "lib" .. name .. ".so")) and
                               not os.isfile(path.join(directory, "lib" .. name .. ".a")) then
                                found = false
                                break
                            end
                        end
                        if found then
                            return directory
                        end
                    end
                end
                return nil
            end

            local tensorrt_include = find_header(tensorrt_roots, {
                "include/NvInfer.h",
                "include/x86_64-linux-gnu/NvInfer.h",
                "include/aarch64-linux-gnu/NvInfer.h",
                "NvInfer.h"
            })
            local cuda_include = find_header(cuda_roots, {
                "include/cuda_runtime_api.h",
                "include/x86_64-linux-gnu/cuda_runtime_api.h",
                "include/aarch64-linux-gnu/cuda_runtime_api.h"
            })
            local tensorrt_lib = find_library_dir(tensorrt_roots, {"nvinfer", "nvonnxparser"})
            local cuda_lib = find_library_dir(cuda_roots, {"cudart"})

            assert(tensorrt_include and tensorrt_lib and cuda_include and cuda_lib,
                "TensorRT/CUDA was not found; pass --tensorrt_root=<prefix> " ..
                "and --cuda_root=<prefix> (or set TENSORRT_ROOT/CUDA_HOME)")
            target:add("includedirs", tensorrt_include, cuda_include, {public = true})
            target:add("linkdirs", tensorrt_lib, cuda_lib, {public = true})
            target:add("rpathdirs", tensorrt_lib, cuda_lib, {public = true})
            target:add("defines", "NEWVISION_HAS_TENSORRT=1", {public = true})
            target:add("links", "nvinfer", "nvonnxparser", "cudart", {public = true})
        end
    end)


-- ---------------------------------------------------------------------------
-- 测试目标按 tests/*.cpp 递归生成，文件名即目标名。新增一个 smoke 只需把文件
-- 放进 tests/，不必再手写一遍 target() 块。
--
-- 默认行为是 add_deps("newvision")。只有两类需要在下面登记：
--   standalone —— 不牵扯相机/串口 SDK，自己列出所需源文件，这样在没有硬件库
--                 的机器上也能单独构建；
--   gated      —— 依赖某个配置开关或平台才存在。
-- ---------------------------------------------------------------------------

local standalone_tests = {
    logger_smoke = {
        files    = {"src/l6_telemetry/logger.cpp"},
        includes = {"include", "tools/logger/include", "tools/logger/include/3rdparty"},
    },
    latest_buffer_smoke = {
        includes = {"tools/LatestBuffer/include"},
    },
    fps_counter_smoke = {
        files    = {"src/l6_telemetry/fps_counter.cpp"},
        includes = {"include"},
    },
    udp_json_sender_smoke = {
        files    = {"src/l6_telemetry/udp_json_sender.cpp"},
        includes = {"include", "tools/logger/include/3rdparty"},
        syslinks = {"pthread"},
    },
    serial_protocol_smoke = {
        files    = {"src/l1_sensor/serial/serial_protocol.cpp", "src/l6_telemetry/logger.cpp"},
        includes = {"include", "tools/logger/include", "tools/logger/include/3rdparty"},
    },
    -- 灯条精修只依赖 OpenCV，不牵扯相机/串口 SDK。
    armor_refiner_smoke = {
        files    = {"src/l2_perception/armor/armor_refiner.cpp"},
        includes = {"include", "/usr/include/eigen3"},
        opencv   = {"opencv_core", "opencv_imgproc"},
    },
    -- 五次多项式过渡段全是标量，刻意不碰 Eigen/OpenCV，这样数值部分
    -- （六系数、三次极值、二分）能在没有相机没有串口的机器上单独验完。
    aim_smoother_smoke = {
        files    = {"src/l4_planning/aim_smoother.cpp"},
        includes = {"include"},
    },
}

-- 需要开关或平台才存在的目标。openvino 这几个都要模型；
-- armor_refiner_video_test 和 auto_aim_test 还要显示器。
local gated_tests = {
    openvino_armor_smoke     = "use_openvino",
    auto_aim_test            = "use_openvino",
    track_diag               = "use_openvino",
    armor_refiner_video_test = "use_openvino",
    serial_worker_smoke      = "linux",
}

-- serial_worker_smoke 用 pty 造串口，需要额外链 util。
local extra_syslinks = {
    serial_worker_smoke = {"util"},
}

-- tests/main.cpp 是完整运行时的入口，目标名不叫 main。
local renamed_tests = {
    main = "auto_aim",
}

for _, source in ipairs(os.files("tests/*.cpp")) do
    local name = renamed_tests[path.basename(source)] or path.basename(source)

    local gate = gated_tests[name]
    local enabled = true
    if gate == "linux" then
        enabled = is_plat("linux")
    elseif gate then
        enabled = has_config(gate)
    end

    if enabled then
        target(name)
            set_kind("binary")
            set_default(false)
            set_rundir("$(projectdir)")
            add_files(source)

            local spec = standalone_tests[name]
            if spec then
                for _, file in ipairs(spec.files or {}) do
                    add_files(file)
                end
                for _, dir in ipairs(spec.includes or {}) do
                    add_includedirs(dir)
                end
                for _, link in ipairs(spec.syslinks or {}) do
                    add_syslinks(link)
                end
                if spec.opencv then
                    if has_config("use_xrepo_deps") then
                        add_packages("opencv")
                    elseif has_config("use_system_deps") then
                        add_includedirs("/usr/include/opencv4")
                        add_links(table.unpack(spec.opencv))
                    end
                end
            else
                add_deps("newvision")
            end

            for _, link in ipairs(extra_syslinks[name] or {}) do
                add_syslinks(link)
            end
        target_end()
    end
end
