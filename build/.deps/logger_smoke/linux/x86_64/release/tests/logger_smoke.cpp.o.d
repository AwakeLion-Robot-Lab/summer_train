{
    depfiles_format = "gcc",
    depfiles = "logger_smoke.o: tests/logger_smoke.cpp include/l6_telemetry/logger.hpp\
",
    values = {
        "/usr/bin/g++-13",
        {
            "-m64",
            "-fvisibility=hidden",
            "-fvisibility-inlines-hidden",
            "-O3",
            "-std=c++20",
            "-Iinclude",
            "-Itools/logger/include",
            "-Itools/logger/include/3rdparty",
            "-DNDEBUG"
        }
    },
    files = {
        "tests/logger_smoke.cpp"
    }
}