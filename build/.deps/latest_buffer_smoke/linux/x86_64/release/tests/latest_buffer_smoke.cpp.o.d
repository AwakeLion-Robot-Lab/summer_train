{
    depfiles_format = "gcc",
    depfiles = "latest_buffer_smoke.o: tests/latest_buffer_smoke.cpp  tools/LatesBuffer/include/latesbuffer.hpp\
",
    files = {
        "tests/latest_buffer_smoke.cpp"
    },
    values = {
        "/usr/bin/g++-13",
        {
            "-m64",
            "-fvisibility=hidden",
            "-fvisibility-inlines-hidden",
            "-O3",
            "-std=c++20",
            "-Itools/LatesBuffer/include",
            "-DNDEBUG"
        }
    }
}