#pragma once
#define LEVEL 4
#define PATTERNNUMFORINDEX 8
#include <vector>
#include "types.h"
using namespace std;

extern vector<int> wG;
extern vector<int> hG;

extern vector<float> fxG;
extern vector<float> fyG;
extern vector<float> cxG;
extern vector<float> cyG;
extern vector<Matrix3d> KG;
extern vector<Matrix3d> KinvG;

extern vector<pair<int, int>> pattern;
extern int patternNum;
extern int huberThreshold;
extern float initializationDropThreshold;
extern float initializationTransformNum;
extern float outlierEnergyThreshold;
extern float setting_idepthFixPrior;