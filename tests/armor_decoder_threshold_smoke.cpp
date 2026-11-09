#include "l1_perception/armor/armor_decoder.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

int main() {
  using namespace L2Perception;
  const float inf = std::numeric_limits<float>::infinity();
  for (float threshold : {0.1F, 0.5F, std::nextafter(0.5F, 1.0F), 0.7F, 1.0F}) {
    for (float logit : {-inf, -100.F, -1.F, -1e-30F, -0.F, 0.F, 1e-30F, 0.84729785F, 1.F, 100.F, inf,
                        std::numeric_limits<float>::quiet_NaN()}) {
      ArmorDecoderConfig config;
      config.confidence_threshold = threshold;
      config.minimum_confidence = -1.F;
      config.nms_score_threshold = 0.F;
      std::vector<float> row{20,20,20,60,70,60,70,20,logit};
      row.resize(22, -5.F); row[9] = row[16] = 10.F;
      InferenceTensor tensor;
      tensor.name = "output"; tensor.shape = {1,1,22}; tensor.setOwnedData(row);
      const auto decoded = ArmorDecoder(config).decode({{tensor}}, {});
      const double x = logit;
      const double score = x > 0 ? 1.0 / (1.0 + std::exp(-x))
                                  : std::exp(x) / (1.0 + std::exp(x));
      const bool expected = std::isfinite(score) && score >= threshold && static_cast<float>(score) > 0;
      if (decoded.size() != static_cast<unsigned>(expected))
        throw std::runtime_error("sigmoid threshold behavior changed");
      if (expected && decoded[0].confidence != static_cast<float>(score))
        throw std::runtime_error("confidence changed");
    }
  }
  std::cout << "decoder: 65 threshold/boundary/nonfinite cases passed\n";
}
