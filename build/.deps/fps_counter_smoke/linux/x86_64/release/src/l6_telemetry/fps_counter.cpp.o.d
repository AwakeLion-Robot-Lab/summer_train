{
    depfiles_format = "gcc",
    files = {
        "src/l6_telemetry/fps_counter.cpp"
    },
    depfiles = "fps_counter.o: src/l6_telemetry/fps_counter.cpp  include/l6_telemetry/fps_counter.hpp\
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
            "-DNDEBUG"
        }
    }
}