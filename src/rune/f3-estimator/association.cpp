// Algorithm reference: Alliance-Algorithm/rmcs_auto_aim_v2 (MIT).
// See ALLIANCE-LICENSE and README.md in this directory.
#include "rune/f3-estimator/association.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace Rune {
std::vector<int> hungarianAssign(const Eigen::MatrixXd& costs, double gate) {
  if (!std::isfinite(gate) || gate<=0)
    throw std::invalid_argument("Hungarian gate must be finite and positive");
  const int rows=costs.rows(), real_cols=costs.cols(), cols=real_cols+rows;
  std::vector<int> result(rows,-1);
  if (!rows || !real_cols) return result;
  // Rectangular shortest augmenting path, with rows <= columns.
  const double huge=gate*(rows+cols+1)+1;
  const auto cost=[&](int i,int j) {
    if (j>=real_cols) return gate;
    const double c=costs(i,j);
    return std::isfinite(c) && c>=0 && c<gate ? c : huge;
  };
  std::vector<double> u(rows+1),v(cols+1);
  std::vector<int> p(cols+1),way(cols+1);
  for (int i=1;i<=rows;++i) {
    p[0]=i;
    int j0=0;
    std::vector<double> minimum(cols+1,std::numeric_limits<double>::infinity());
    std::vector<bool> used(cols+1,false);
    do {
      used[j0]=true;
      const int i0=p[j0];
      double delta=std::numeric_limits<double>::infinity();
      int j1=0;
      for (int j=1;j<=cols;++j) if (!used[j]) {
        const double current=cost(i0-1,j-1)-u[i0]-v[j];
        if (current<minimum[j]) { minimum[j]=current; way[j]=j0; }
        if (minimum[j]<delta) { delta=minimum[j]; j1=j; }
      }
      for (int j=0;j<=cols;++j) {
        if (used[j]) { u[p[j]]+=delta; v[j]-=delta; }
        else minimum[j]-=delta;
      }
      j0=j1;
    } while (p[j0]!=0);
    do { const int j1=way[j0]; p[j0]=p[j1]; j0=j1; } while (j0);
  }
  for (int j=1;j<=real_cols;++j)
    if (p[j] && cost(p[j]-1,j-1)<gate) result[p[j]-1]=j-1;
  return result;
}
} // namespace Rune
