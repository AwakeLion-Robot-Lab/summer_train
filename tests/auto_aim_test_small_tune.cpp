#include "rune_replay.hpp"

int main(int argc, char** argv) {
  return runRuneReplay(argc, argv, Rune::Mode::Small, "auto_aim_test_small_tune",
                       "video/rune/2026-10-28_00-05-41.avi");
}
