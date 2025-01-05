#pragma once
#include "frame.h"
#include "types.h"
#include "point.h"
#include "trajectory.h"
#include <algorithm>

class Trajectory;

class SingleTracker
{
public:
    SingleTracker();
    ~SingleTracker();
    void setRefFrame(shared_ptr<Frame> _refFrame);
    void setTarFrame(shared_ptr<Frame> _tarFrame);
    void setTrajectory(shared_ptr<Trajectory> _trajectoryer);
    Vector2f incrementalEquationLevel(int level);
    Vector2f wrongIncrementalEquationLevel(int level, SE3 initialPosePredicted);
    void probeDepthThisLevel(int level, float lambda, Vector8f incrementPoseAffineThisLevel);
    Vector2f checkUpdateLevel(int level);
    void updateDepthThisLevel(int level);
    void spreadDepthToNextLevel(int nextLevel);
    void optimizeRelativePose();
    void wrongOptimizeRelativePose();
    void exchangeOriginalAndNew();
    void smoothDepth(int level);
    void resetPoints(int level);
    
    shared_ptr<Trajectory> trajectory;

    shared_ptr<Frame> refFrame;
    shared_ptr<Frame> tarFrame;
    vector<int> numPoints;
    Eigen::DiagonalMatrix<float, 8> weightEquation;
    
    vector<Matrix8f> HSchur;
    vector<Vector8f> bSchur;
    vector<Matrix8f> HPoseAffine;
    vector<Vector8f> bPoseAffine;

    vector<Matrix8f> HSchurNew;
    vector<Vector8f> bSchurNew;
    vector<Matrix8f> HPoseAffineNew;
    vector<Vector8f> bPoseAffineNew;

    vector<Vector8f> JAlpha_mul_JBeta2ml;
    vector<float> bAlpha2ml;
    vector<float> JAlpha_mul_JAlphaSingle2ml;

    vector<Vector8f> JAlpha_mul_JBeta2mlNew;
    vector<float> bAlpha2mlNew;
    vector<float> JAlpha_mul_JAlphaSingle2mlNew;

    vector<Matrix8f> H;
    vector<Vector8f> b;

    SE3 relativaPose;
    Vector2d relativeAffine;
    SE3 relativaPoseNew;
    Vector2d relativeAffineNew;
    SE3 relativePosePredictedByUniformSpeed;



    int debugSign;



    void optimizeDSO();
    void spreadDepthToNextLevelDSO(int nextLevel);
    Vector3f incrementalEquationLevelDSO(int level, const SE3& _poseRefToTarCurrent, AffineLight _affRefToTarCurrent);
    void probeDepthThisLevelDSO(int level, float lambda, Vector8f incrementPoseAffineThisLevel);
    Vector3f incrementalEquationLevelFollowUpDSO(int level, const SE3& _poseRefToTarCurrent, AffineLight _affRefToTarCurrent);
    Vector3f incrementalEquationDSO(int level, Matrix8f& HS, Vector8f& bS, Matrix8f& HP, Vector8f& bP, const SE3& _poseRefToTarCurrent, AffineLight _affRefToTarCurrent);
    Vector2f compareEnergyRegression(int level);
    void exchangeOriginAndNewDSO();
    void updateDepthThisLevelDSO(int level);
    void climbDepthLevelDSO(int level);
    float alphaK;
    float alphaW;
    float regWeight;
    float couplingWeight;
    bool mutation;
    int positionMutation;
    vector<int> maxIterations;


    
    void optimizeRelativePose2ml();
    void spreadDepthToNextLevel2ml(int nextLevel);
    Vector3f incrementalEquation2ml(int level);
    void probeDepth2ml(int level, float lambda, Vector8f incrementPoseAffineThisLevel);
    Vector3f checkEnergy2ml(int level);
    void updateDepthThisLevel2ml(int level);
    void climbDepthLevel2ml(int level);
    void smoothDepth2ml(int level);
    Vector2f compareEnergyRegression2ml(int level);


};