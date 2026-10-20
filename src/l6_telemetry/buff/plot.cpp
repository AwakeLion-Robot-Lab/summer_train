#include "l6_telemetry/buff/plot.hpp"

#include <iomanip>
#include <sstream>

namespace L6Telemetry::Buff {

std::string csvHeader()
{
  return "timestamp,x,y,z,rotation_speed,rotation_angle,face_yaw,blade,valid,degraded,yaw,pitch,shoot\n";
}

std::string toCsv(const PlotSample& sample)
{
  const double timestamp = std::chrono::duration<double>(
    sample.timestamp.time_since_epoch()).count();
  std::ostringstream output;
  output << std::setprecision(12)
         << timestamp << ',' << sample.state.x << ',' << sample.state.y << ','
         << sample.state.z << ',' << sample.state.rotation_speed << ','
         << sample.state.rotation_angle << ',' << sample.state.face_yaw << ','
         << sample.plan.blade_index << ',' << sample.plan.valid << ','
         << sample.plan.degraded << ',' << sample.plan.yaw << ','
         << sample.plan.pitch << ',' << sample.plan.algorithm_shoot << '\n';
  return output.str();
}

}  // namespace L6Telemetry::Buff
