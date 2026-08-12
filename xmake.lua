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

option("use_gtsam")
    set_default(false)
    set_showmenu(true)
    set_description("Enable the optional GTSAM factor-graph estimator (l3_estimation/gtsam_est)")
option_end()

-- GTSAM 的安装前缀。默认 /usr/local（源码安装的惯例位置）；装在别处时用
-- `xmake f --gtsam_root=$HOME/.local/gtsam` 指过去。做成 option 而不是读环境
-- 变量：option 会被 xmake f 持久化到 .xmake/，环境变量则要求每次 build 都
-- export，忘了就会静默回退到 /usr/local 并链到另一份 GTSAM 上。
option("gtsam_root")
    set_default("/usr/local")
    set_showmenu(true)
    set_description("GTSAM install prefix (used only when use_gtsam=y)")
option_end()

if has_config("use_xrepo_deps") then
    add_requires("opencv", {optional = true})
    add_requires("yaml-cpp", {optional = true})
end

target("newvision")
    set_kind("static")
    add_files("src/**/*.cpp")
    -- daedalus_auto_aim.cpp and daedalus_noise_calib.cpp own main() and are
    -- compiled only by their binary targets. The shared-memory source itself
    -- is Linux-only.
    remove_files("src/runtime/daedalus_auto_aim.cpp")
    remove_files("src/runtime/daedalus_noise_calib.cpp")
    if not is_plat("linux") then
        remove_files("src/l1_sensor/daedalus_source.cpp")
        remove_files("src/l1_sensor/daedalus_ground_truth.cpp")
    end
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
    -- 可选依赖统一在**一个** on_load 里处理。xmake 的 on_xxx 是覆盖语义，写成
    -- 两个 on_load 会让后一个顶掉前一个：同时开 use_openvino 和 use_gtsam 时
    -- NEWVISION_HAS_OPENVINO 会静默消失，OpenVINO 后端被无声禁用。
    on_load(function (target)
        if has_config("use_openvino") then
            -- 推理结果的最后几个 ulp 会随 OpenVINO 版本变化，而 1 度步长的离散 yaw
            -- 搜索会把这种差异放大成不同的选中角度，所以优先锁定本机安装的
            -- 2024.6.0；没有这个版本的机器再回退 pkg-config。
            local openvino_runtime = "/opt/intel/openvino_2024.6.0/runtime"
            local openvino_include = path.join(openvino_runtime, "include")
            local openvino_lib = path.join(openvino_runtime, "lib", "intel64")
            if os.isdir(openvino_include) and os.isfile(path.join(openvino_lib, "libopenvino.so")) then
                target:add("includedirs", openvino_include)
                target:add("linkdirs", openvino_lib, {public = true})
                target:add("rpathdirs", openvino_lib, {public = true})
            else
                import("lib.detect.find_package")
                local openvino = find_package("pkgconfig::openvino", {version = true})
                assert(openvino,
                    "OpenVINO was not found; install 2024.6 or expose it through pkg-config")
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
            -- TensorRT/CUDA 没有统一的 pkg-config 布局。优先使用显式 option，
            -- 其次读取 TENSORRT_ROOT/CUDA_HOME，最后覆盖 Ubuntu 的常见系统路径。
            local function append_unique(list, value)
                if value == nil or value == "" then
                    return
                end
                for _, existing in ipairs(list) do
                    if existing == value then
                        return
                    end
                end
                table.insert(list, value)
            end

            local trt_roots = {}
            append_unique(trt_roots, get_config("tensorrt_root"))
            append_unique(trt_roots, os.getenv("TENSORRT_ROOT"))
            append_unique(trt_roots, "/usr/local/TensorRT")
            append_unique(trt_roots, "/usr/local")
            append_unique(trt_roots, "/usr")

            local cuda_roots = {}
            append_unique(cuda_roots, get_config("cuda_root"))
            append_unique(cuda_roots, os.getenv("CUDA_HOME"))
            append_unique(cuda_roots, os.getenv("CUDA_PATH"))
            append_unique(cuda_roots, "/usr/local/cuda")
            append_unique(cuda_roots, "/usr")

            local trt_include
            for _, root in ipairs(trt_roots) do
                local candidates = {
                    path.join(root, "include", "NvInfer.h"),
                    path.join(root, "include", "x86_64-linux-gnu", "NvInfer.h"),
                    path.join(root, "include", "aarch64-linux-gnu", "NvInfer.h"),
                    path.join(root, "NvInfer.h")
                }
                for _, candidate in ipairs(candidates) do
                    if os.isfile(candidate) then
                        trt_include = path.directory(candidate)
                        break
                    end
                end
                if trt_include then
                    break
                end
            end

            local function find_library_dir(roots, required_names)
                for _, root in ipairs(roots) do
                    local directories = {
                        path.join(root, "lib"),
                        path.join(root, "lib64"),
                        path.join(root, "lib", "x86_64-linux-gnu"),
                        path.join(root, "lib", "aarch64-linux-gnu")
                    }
                    for _, directory in ipairs(directories) do
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

            local trt_lib = find_library_dir(trt_roots, {"nvinfer", "nvonnxparser"})
            local cuda_include
            for _, root in ipairs(cuda_roots) do
                local candidate = path.join(root, "include", "cuda_runtime_api.h")
                if os.isfile(candidate) then
                    cuda_include = path.directory(candidate)
                    break
                end
            end
            local cuda_lib = find_library_dir(cuda_roots, {"cudart"})

            assert(trt_include and trt_lib and cuda_include and cuda_lib,
                "TensorRT/CUDA was not found; pass --tensorrt_root=<prefix> " ..
                "and --cuda_root=<prefix> (or set TENSORRT_ROOT/CUDA_HOME)")
            target:add("includedirs", trt_include, cuda_include, {public = true})
            target:add("linkdirs", trt_lib, cuda_lib, {public = true})
            target:add("rpathdirs", trt_lib, cuda_lib, {public = true})
            target:add("defines", "NEWVISION_HAS_TENSORRT=1", {public = true})
            target:add("links", "nvinfer", "nvonnxparser", "cudart", {public = true})
        end

        if has_config("use_gtsam") then
            -- 因子图估计器（l3_estimation/gtsam_est）。GTSAM 一般是源码装到
            -- /usr/local，没有 pkg-config 文件，所以这里直接按安装前缀找；
            -- 布局和 jlu_vision_26-master 的 rule("gtsam_deps") 一致。
            -- 关掉这个选项时 gtsam_est/tracker.cpp 照常编译，只是构造时抛异常，
            -- 整棵树在没装 GTSAM 的机器上仍然可以 build。
            local prefix = get_config("gtsam_root") or "/usr/local"
            local include = path.join(prefix, "include")
            local lib = path.join(prefix, "lib")
            assert(os.isdir(path.join(include, "gtsam")),
                "GTSAM was not found under " .. prefix ..
                "; install it or pass --gtsam_root=<prefix> (see docs/gtsam_est_port.md)")
            -- gtsam_est/target.hpp 和 factors.hpp 是可测试的公共头，依赖目标需要继承
            -- GTSAM include 路径；否则库本身能编，独立烟雾测试却找不到头文件。
            target:add("includedirs", include, {public = true})
            target:add("linkdirs", lib, {public = true})
            target:add("rpathdirs", lib, {public = true})
            target:add("defines", "NEWVISION_USE_GTSAM=1", {public = true})
            target:add("links", "gtsam", {public = true})
            -- metis / cephes 是 GTSAM 自带的第三方库，按构建配置决定是否单独成库；
            -- 动态库版本会由 libgtsam.so 的 DT_NEEDED 自动带出来，所以只在真的
            -- 存在时才显式链接，避免在没有它们的安装上链接失败。
            for _, name in ipairs({"metis-gtsam", "cephes-gtsam"}) do
                if os.isfile(path.join(lib, "lib" .. name .. ".a")) or
                   os.isfile(path.join(lib, "lib" .. name .. ".so")) then
                    target:add("links", name, {public = true})
                end
            end
            -- TBB 同理：GTSAM 若开了 TBB，动态库会自己带上依赖。本项目是单线程
            -- 流水线，建议 GTSAM 直接 -DGTSAM_WITH_TBB=OFF，见 docs/gtsam_est_port.md。
        end
    end)

target("auto_aim")
    set_kind("binary")
    set_default(false)
    set_rundir("$(projectdir)")
    add_files("tests/main.cpp")
    add_deps("newvision")

if is_plat("linux") then
target("daedalus_auto_aim")
    set_kind("binary")
    set_default(false)
    set_rundir("$(projectdir)")
    add_files("src/runtime/daedalus_auto_aim.cpp")
    add_deps("newvision")

-- 观测噪声标定需要至少一个真正可用的推理后端。
if has_config("use_openvino") or has_config("use_tensorrt") then
target("daedalus_noise_calib")
    set_kind("binary")
    set_default(false)
    set_rundir("$(projectdir)")
    add_files("src/runtime/daedalus_noise_calib.cpp")
    add_deps("newvision")
end

target("daedalus_source_smoke")
    set_kind("binary")
    set_default(false)
    set_rundir("$(projectdir)")
    add_files("tests/daedalus_source_smoke.cpp")
    add_deps("newvision")

target("pnp_truth_residual_smoke")
    set_kind("binary")
    set_default(false)
    set_rundir("$(projectdir)")
    add_files("tests/pnp_truth_residual_smoke.cpp")
    add_deps("newvision")

target("daedalus_ground_truth_smoke")
    set_kind("binary")
    set_default(false)
    set_rundir("$(projectdir)")
    add_files("tests/daedalus_ground_truth_smoke.cpp")
    -- 真值通道只依赖 Eigen 和 POSIX 共享内存，不牵扯相机/串口 SDK。
    add_files("src/l1_sensor/daedalus_ground_truth.cpp")
    add_includedirs("include")
    add_includedirs("/usr/include/eigen3")
end

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

target("tensorrt_backend_smoke")
    set_kind("binary")
    set_default(false)
    set_rundir("$(projectdir)")
    add_files("tests/tensorrt_backend_smoke.cpp")
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

target("estimator_backend_smoke")
    set_kind("binary")
    set_default(false)
    set_rundir("$(projectdir)")
    add_files("tests/estimator_backend_smoke.cpp")
    add_deps("newvision")

if has_config("use_gtsam") then
target("gtsam_est_smoke")
    set_kind("binary")
    set_default(false)
    set_rundir("$(projectdir)")
    add_files("tests/gtsam_est_smoke.cpp")
    add_deps("newvision")
end

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
