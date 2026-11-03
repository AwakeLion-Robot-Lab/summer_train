#include "rune_replay.hpp"

int main(int argc, char** argv) {
  return runRuneReplay(argc, argv, Rune::Mode::Big, "auto_aim_test_big_tune",
                       "video/rune/2026-10-28_00-03-18.avi");
}
