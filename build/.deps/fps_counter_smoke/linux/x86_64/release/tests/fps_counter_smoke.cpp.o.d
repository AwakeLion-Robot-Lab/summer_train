{
    depfiles_format = "gcc",
    files = {
        "tests/fps_counter_smoke.cpp"
    },
    depfiles = "fps_counter_smoke.o: tests/fps_counter_smoke.cpp  include/l6_telemetry/fps_counter.hpp\
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