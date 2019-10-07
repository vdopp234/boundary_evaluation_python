#ifndef EDGE_EVAL_H
#define EDGE_EVAL_H

namespace edge_eval {

struct returnVal {
    double* out1;
    double* out2;
    double cost;
    double oc;
};

class Eval
{
  public:
    returnVal correspondPixels(double* bmap1, double* bmap2, int rows, int cols,
    double maxDist = 0.0075, double outlierCost = 100);

}; // EDGE_EVAL

} // namespace edge_eval

#endif
