#include "global_params.h"

vector<int> wG;
vector<int> hG;

vector<float> fxG;
vector<float> fyG;
vector<float> cxG;
vector<float> cyG;
vector<Matrix3d> KG;
vector<Matrix3d> KinvG;

vector<pair<int, int>> pattern;
int patternNum;
int huberThreshold;
float initializationDropThreshold;