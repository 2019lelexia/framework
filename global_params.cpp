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
float initializationTransformNum = 1500;
float outlierEnergyThreshold = 12 * 12;
float setting_idepthFixPrior = 50 * 50;
