#include "rune_replay.hpp"

int main(int argc, char** argv) {
  return runRuneReplay(argc, argv, Rune::Mode::Big, "auto_aim_test_big_rune",
                       "video/rune/2026-11-15_15-05-03.mkv");
}
