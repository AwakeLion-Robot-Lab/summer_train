{
    depfiles_format = "gcc",
    depfiles = "latency_compensator.o: src/l4_planning/latency_compensator.cpp  include/l4_planning/latency_compensator.hpp\
",
    files = {
        "src/l4_planning/latency_compensator.cpp"
    },
    values = {
        "/usr/bin/g++-13",
        {
            "-m64",
            "-fvisibility=hidden",
            "-fvisibility-inlines-hidden",
            "-O3",
            "-std=c++20",
            "-Iinclude",
            "-Itools/config_set/include",
            "-Itools/LatesBuffer/include",
            "-Itools/camera_sdk",
            "-Itools/camera_sdk/hikrobot",
            "-Itools/camera_sdk/hikrobot/include",
            "-Itools/camera_sdk/mindvision",
            "-Itools/camera_sdk/mindvision/include",
            "-Itools/logger/include",
            "-Itools/logger/include/3rdparty",
            "-I/usr/include/opencv4",
            "-DNDEBUG"
        }
    }
}