#include "rune_replay.hpp"

int main(int argc, char** argv) {
  return runRuneReplay(argc, argv, Rune::Mode::Small, "auto_aim_test_small_rune",
                       "video/rune/2026-11-15_15-04-07.mkv");
}
