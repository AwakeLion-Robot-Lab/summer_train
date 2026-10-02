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

option("openvino_root")
    set_default("")
    set_showmenu(true)
    set_description("OpenVINO install prefix (the directory containing runtime/); empty = auto-detect")
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
    -- OpenVINO 与 TensorRT 共用一个 on_load；xmake 对同一 target 的多个 on_load
    -- 使用覆盖语义，拆开写会让其中一个后端静默失效。
    on_load(function (target)
        if has_config("use_openvino") then
            -- 每台电脑的 OpenVINO 装在哪、装的哪一版都不一样，按下面的顺序找，
            -- 第一个头文件和库都齐全的就用：
            --   1. xmake f --openvino_root=<安装目录>（写错只警告，接着往下找）；
            --   2. 环境变量 INTEL_OPENVINO_DIR（source setupvars.sh 之后就有）；
            --   3. /opt/intel/openvino_2024.6.0：SP-Vision 固定用这一版。推理结果
            --      最后几个 ulp 随 Runtime 版本变，SP 的 1 度离散 yaw 搜索会放大这种
            --      差异，所以本机有这一版就和 SP 链同一版；
            --   4. /opt/intel/openvino（官方安装说明里的软链），再是其余
            --      /opt/intel/openvino*，按目录名倒序，新版本在前；
            --   5. apt 装的系统版（/usr、/usr/local）；
            --   6. pkg-config。
            -- 压缩包装法的库在 runtime/lib/<arch>，arch 按目标架构取 intel64 或
            -- aarch64：同一台机器上可能放着别的架构的包，只比版本号会挑中链不上的。
            -- 用的是哪一份，xmake -v 会打印；换版本后离线基线都要重跑，ulp 级差异
            -- 会一路传到 pred_px。
            local ov_arch = target:is_arch("arm64", "aarch64") and "aarch64" or "intel64"
            local ov_triplet = ov_arch == "aarch64" and "aarch64-linux-gnu" or "x86_64-linux-gnu"
            local sp_root = "/opt/intel/openvino_2024.6.0"

            -- 压缩包装法：<root>/runtime/{include,lib/<arch>}。
            local function find_archive(root)
                if root == nil or root == "" then
                    return nil
                end
                local include = path.join(root, "runtime", "include")
                local lib = path.join(root, "runtime", "lib", ov_arch)
                if os.isfile(path.join(include, "openvino", "openvino.hpp")) and
                   os.isfile(path.join(lib, "libopenvino.so")) then
                    return {root = root, include = include, lib = lib}
                end
            end

            -- 系统版在编译器和链接器的默认搜索路径里，不往 includedirs 里加：
            -- -I/usr/include 会打乱 libstdc++ 的 #include_next <stdlib.h>。
            local function find_system(prefix)
                if not os.isfile(path.join(prefix, "include", "openvino", "openvino.hpp")) then
                    return nil
                end
                for _, sub in ipairs({"lib/" .. ov_triplet, "lib64", "lib"}) do
                    if os.isfile(path.join(prefix, sub, "libopenvino.so")) then
                        return {root = prefix}
                    end
                end
            end

            local tried = {}
            local function try_archive(root)
                if root == nil or root == "" or tried[root] then
                    return nil
                end
                tried[root] = true
                return find_archive(root)
            end

            local configured = get_config("openvino_root") or ""
            local found = try_archive(configured)
            if not found and configured ~= "" and configured ~= sp_root then
                cprint("${color.warning}openvino_root=%s 下没有 %s 的 OpenVINO，改为自动查找",
                    configured, ov_arch)
            end
            found = found or try_archive(os.getenv("INTEL_OPENVINO_DIR"))
                or try_archive(sp_root)
                or try_archive("/opt/intel/openvino")
            if not found then
                local others = os.dirs("/opt/intel/openvino*")
                table.sort(others, function (a, b) return a > b end)
                for _, root in ipairs(others) do
                    found = try_archive(root)
                    if found then
                        break
                    end
                end
            end
            found = found or find_system("/usr") or find_system("/usr/local")

            if found then
                vprint("OpenVINO: %s", found.root)
                if found.include then
                    target:add("includedirs", found.include)
                    target:add("linkdirs", found.lib, {public = true})
                    target:add("rpathdirs", found.lib, {public = true})
                end
            else
                import("lib.detect.find_package")
                local openvino = find_package("pkgconfig::openvino", {version = true})
                assert(openvino,
                    "OpenVINO was not found (" .. ov_arch .. "); pass --openvino_root=<prefix>, " ..
                    "source setupvars.sh, or expose it through pkg-config")
                vprint("OpenVINO: pkg-config %s", openvino.version or "")
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
        includes = {"tools/LatesBuffer/include"},
    },
    udp_json_sender_smoke = {
        files    = {"src/l6_telemetry/udp_json_sender.cpp"},
        includes = {"include", "tools/logger/include/3rdparty"},
        syslinks = {"pthread"},
    },
    daedalus_client_smoke = {
        files    = {"src/l1_sensor/simulator/daedalus_client.cpp"},
        includes = {"include"},
        opencv   = {"opencv_core", "opencv_imgproc"},
        syslinks = {"pthread"},
    },
    serial_protocol_smoke = {
        files    = {"src/l1_sensor/serial/serial_protocol.cpp", "src/l6_telemetry/logger.cpp"},
        includes = {"include", "tools/logger/include", "tools/logger/include/3rdparty"},
    },
    -- SO(3) 是 header-only，只依赖 Eigen 与 ceres/jet.h（后者也是纯头文件，
    -- 不需要链 libceres）。
    so3_smoke = {
        includes = {"include", "/usr/include/eigen3"},
    },
    -- 整车模型只依赖 Eigen 与 L3 的类型定义，同样不牵扯 SDK。
    vehicle_model_smoke = {
        includes = {"include", "/usr/include/eigen3"},
        opencv   = {"opencv_core"},
    },
    error_state_ekf_smoke = {
        includes = {"include", "/usr/include/eigen3"},
        opencv   = {"opencv_core"},
    },
    -- 端点观测要 calib3d 里的 projectPoints 做对拍参照。
    light_measure_smoke = {
        includes = {"include", "/usr/include/eigen3"},
        opencv   = {"opencv_core", "opencv_calib3d"},
    },
}

-- 需要开关或平台才存在的目标。openvino 这几个都要模型；auto_aim_test 还要显示器。
local gated_tests = {
    openvino_armor_smoke     = "use_openvino",
    auto_aim_test            = "use_openvino",
    track_diag               = "use_openvino",
    light_noise              = "use_openvino",
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

-- 相机内参标定，来自 atooooooom 分支，用法见 tools/camera_calibration/README.md。
-- 源文件按 "tools/camera_calibration/..." 引用，只给这两个目标加项目根目录。
target("camera_capture")
    set_kind("binary")
    set_default(false)
    set_rundir("$(projectdir)")
    add_files("tools/camera_calibration/camera_capture.cpp")
    add_files("tools/camera_calibration/timed_image_saver.cpp")
    add_includedirs("$(projectdir)")
    add_deps("newvision")
target_end()

target("camera_calibrator")
    set_kind("binary")
    set_default(false)
    set_rundir("$(projectdir)")
    add_files("tools/camera_calibration/camera_calibrator.cpp")
    add_files("tools/camera_calibration/high_precision_calibrator.cpp")
    add_includedirs("$(projectdir)")
    add_deps("newvision")
target_end()

target("daedalus_client")
    set_kind("binary")
    set_default(false)
    set_rundir("$(projectdir)")
    add_files("examples/daedalus_client.cpp")
    add_files("src/l1_sensor/simulator/daedalus_client.cpp")
    add_includedirs("include")
    add_syslinks("pthread")
    if has_config("use_xrepo_deps") then
        add_packages("opencv")
    elseif has_config("use_system_deps") then
        add_includedirs("/usr/include/opencv4")
        add_links("opencv_core", "opencv_imgproc", "opencv_highgui")
    end
target_end()

-- Daedalus 实时整车预测可视化。复用完整 newvision 的 L2~L6 链路；
-- OpenVINO/TensorRT 后端由 auto_aim.yaml 与对应 xmake 开关选择。
target("daedalus_vehicle_prediction")
    set_kind("binary")
    set_default(false)
    set_rundir("$(projectdir)")
    add_files("examples/daedalus_vehicle_prediction.cpp")
    add_deps("newvision")
target_end()
