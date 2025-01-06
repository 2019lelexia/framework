#include "tracker.h"

void SingleTracker::optimizeDSO()
{
    alphaK = 2.5 * 2.5;
    alphaW = 150 * 150;
    regWeight = 0.8;
    couplingWeight = 1;

    if(!mutation)
    {
        relativaPose.translation().setZero();
        for(int i = 0; i < LEVEL; i++)
        {
            for(auto iter = refFrame->concernNormalPoints.at(i).begin(); iter != refFrame->concernNormalPoints.at(i).end(); iter++)
            {
                shared_ptr<Point> ptr_point = (*iter);
                ptr_point->depthConvergence = 1;
                ptr_point->depthNew = 1;
                ptr_point->variance = 0;
                ptr_point->JAlphaT_mul_JAlphaSingle = 0;
                ptr_point->JAlphaT_mul_JBeta.setZero();
                ptr_point->bAlpha = 0;
            }
        }
    }

    SE3 poseRefToTarCurrent = relativaPose;
    AffineLight affRefToTarCurrent = AffineLight(relativeAffine[0], relativeAffine[1]);

    Vector3f lastResult = Vector3f::Zero();
    for(int level = LEVEL - 1; level >= 0; level--)
    {
        if(level < LEVEL - 1)
        {
            spreadDepthToNextLevelDSO(level + 1);
        }
        
        Matrix8f Hsc, HP;
        Vector8f bsc, bP;
        resetPoints(level);
        
        
        Vector3f energyLast = incrementalEquationDSO(level, Hsc, bsc, HP, bP, poseRefToTarCurrent, affRefToTarCurrent);
        updateDepthThisLevelDSO(level);
        
        int iterations = 0;
        int failedAttempt = 0;
        float lambda = 0.1;
        float eps = 1e-4;
        int numPointThisLevel = refFrame->concernNormalPoints.at(level).size();
        // cout << "H: " << endl;
        // cout << HPoseAffine.at(level).matrix() << endl;
        // cout << "b: " << endl;
        // cout << bPoseAffine.at(level).matrix() << endl;
        // cout << "Hsc: " << endl;
        // cout << HSchur.at(level).matrix() << endl;
        // cout << "bsc: " << endl;
        // cout << bSchur.at(level).matrix() << endl;
        while(true)
        {
            Matrix8f HThisLevel = HP;
            for(int j = 0; j < 8; j++)
            {
                HThisLevel(j, j) *= (1 + lambda);
            }
            HThisLevel -= Hsc * (1 / (1 + lambda));
            Vector8f bThisLevel = bP - bsc * (1 / (1 + lambda));
            HThisLevel = weightEquation * HThisLevel * weightEquation * (0.01 / (wG[level] * hG[level]));
            bThisLevel = weightEquation * bThisLevel * (0.01 / (wG[level] * hG[level]));

            Vector8f incrementPoseAffine;
            incrementPoseAffine.head<6>() = -(weightEquation.toDenseMatrix().topLeftCorner<6, 6>() * HThisLevel.topLeftCorner<6, 6>().ldlt().solve(bThisLevel.head<6>()));
            incrementPoseAffine.tail<2>().setZero();
            cout << "increment: " << endl;
            cout << incrementPoseAffine.matrix() << endl;
            
            SE3 poseRefToTarNew = SE3::exp(incrementPoseAffine.head<6>().cast<double>()) * poseRefToTarCurrent;
            AffineLight affRefToTarNew = affRefToTarCurrent;
            affRefToTarNew.a += incrementPoseAffine[6];
            affRefToTarNew.b += incrementPoseAffine[7];
            probeDepthThisLevelDSO(level, lambda, incrementPoseAffine);
            // exit(1);
            

            Matrix8f HscNew, HPNew;
            Vector8f bscNew, bPNew;
            Vector3f energyNew = incrementalEquationDSO(level, HscNew, bscNew, HPNew, bPNew, poseRefToTarNew, affRefToTarNew);
            Vector2f energyReg = compareEnergyRegression(level);
            float energyTotallyNew = energyNew[0] + energyNew[1] + energyReg[1];
            float energyTotallyCurrent = energyLast[0] + energyLast[1] + energyReg[0];
            // cout << "energyCurrent: " << energyLast.transpose() << endl;
            // cout << "energyNew: " << energyNew.transpose() << endl;
            cout << "current: " << energyTotallyCurrent << ", new:" << energyTotallyNew << endl; 

            bool approval = energyTotallyNew < energyTotallyCurrent;
            
            if(approval)
            {
                if(energyNew[1] == alphaK * numPointThisLevel)
                {
                    mutation = true;
                }
                Hsc = HscNew;
                bsc = bscNew;
                HP = HPNew;
                bP = bPNew;
                energyLast = energyNew;
                poseRefToTarCurrent = poseRefToTarNew;
                affRefToTarCurrent = affRefToTarNew;
                updateDepthThisLevelDSO(level);
                smoothDepth(level);
                lambda *= 0.5;
                failedAttempt = 0;
                if(lambda < 0.0001)
                {
                    lambda = 0.0001;
                }
            }
            else
            {
                failedAttempt++;
                lambda *= 4;
                if(lambda > 10000)
                {
                    lambda = 10000;
                }
            }
            bool endIteration = false;
            if(!(incrementPoseAffine.norm() > eps) || iterations > maxIterations[level] || failedAttempt >= 2)
            {
                cout << "first:" << incrementPoseAffine.norm() << ", iterations:" << iterations << ", failedAttempt:" << failedAttempt << endl;
                endIteration = true;
            }
            
            if(endIteration)
            {
                break;
            }
            iterations++;
            // cout << "doing: " << iterations << endl;
        }
        lastResult = energyLast;
        // exit(1);
    }
    relativaPose = poseRefToTarCurrent;
    relativeAffine[0] = affRefToTarCurrent.a;
    relativeAffine[1] = affRefToTarCurrent.b;
    cout << relativaPose.matrix() << endl;
    if(!mutation)
    {
        positionMutation = 0;
    }
    else
    {
        positionMutation += 1;
    }
    for(int level = 0; level < LEVEL - 1; level++)
    {
        climbDepthLevelDSO(level);
    }
}

void SingleTracker::climbDepthLevelDSO(int level)
{
    for(auto iter = refFrame->concernNormalPoints.at(level + 1).begin(); iter != refFrame->concernNormalPoints.at(level + 1).end(); iter++)
    {
        shared_ptr<Point> ptr_point = (*iter);
        ptr_point->depthConvergence = 1;
        ptr_point->childNum = 0;
    }
    for(auto iter = refFrame->concernNormalPoints.at(level).begin(); iter != refFrame->concernNormalPoints.at(level).end(); iter++)
    {
        shared_ptr<Point> ptr_point = (*iter);
        if(!ptr_point->isGood)
        {
            continue;
        }
        else
        {
            shared_ptr<Point> ptr_parent = refFrame->concernNormalPoints.at(level + 1).at(ptr_point->indexParent);
            ptr_parent->depthConvergence += ptr_point->depthConvergence * ptr_point->variance;
            ptr_parent->childNum += ptr_point->variance;
        }
    }
    for(auto iter = refFrame->concernNormalPoints.at(level + 1).begin(); iter != refFrame->concernNormalPoints.at(level + 1).end(); iter++)
    {
        shared_ptr<Point> ptr_point = (*iter);
        if(ptr_point->childNum > 0)
        {
            ptr_point->depth = ptr_point->depthConvergence = (ptr_point->depthConvergence / ptr_point->childNum);
            ptr_point->isGood = true;
        }
    }
    smoothDepth(level + 1);
}

void SingleTracker::updateDepthThisLevelDSO(int level)
{
    for(auto iter = refFrame->concernNormalPoints.at(level).begin(); iter != refFrame->concernNormalPoints.at(level).end(); iter++)
    {
        shared_ptr<Point> ptr_point = (*iter);
        if(!ptr_point->isGood)
        {
            ptr_point->depth = ptr_point->depthNew = ptr_point->depthConvergence;
            continue;
        }
        ptr_point->energy = ptr_point->energyNew;
        ptr_point->isGood = ptr_point->isGoodNew;
        ptr_point->depth = ptr_point->depthNew;
        ptr_point->variance = ptr_point->varianceNew;
        ptr_point->bAlpha = ptr_point->bAlphaNew;
        ptr_point->JAlphaT_mul_JAlphaSingle = ptr_point->JAlphaT_mul_JAlphaSingleNew;
        ptr_point->JAlphaT_mul_JBeta = ptr_point->JAlphaT_mul_JBetaNew;
    }
    std::swap(JAlpha_mul_JAlphaSingle2ml, JAlpha_mul_JAlphaSingle2mlNew);
    std::swap(JAlpha_mul_JBeta2ml, JAlpha_mul_JBeta2mlNew);
    std::swap(bAlpha2ml, bAlpha2mlNew);
}

void SingleTracker::exchangeOriginAndNewDSO()
{
    HSchur = HSchurNew;
    bSchur = bSchurNew;
    HPoseAffine = HPoseAffineNew;
    bPoseAffine = bPoseAffineNew;
}

void SingleTracker::spreadDepthToNextLevelDSO(int nextLevel)
{
    for(auto iter = refFrame->concernNormalPoints.at(nextLevel - 1).begin(); iter != refFrame->concernNormalPoints.at(nextLevel - 1).end(); iter++)
    {
        shared_ptr<Point> ptr_point = (*iter);
        shared_ptr<Point> ptr_parent = refFrame->concernNormalPoints.at(nextLevel).at(ptr_point->indexParent);
        if(ptr_parent->isGood == false || ptr_parent->variance < 0.1)
        {
            continue;
        }
        else
        {
            if(ptr_point->isGood == false)
            {
                ptr_point->depthConvergence = ptr_point->depth = ptr_point->depthNew = ptr_parent->depthConvergence;
                ptr_point->isGood = true;
                ptr_point->variance = 0;
            }
            else
            {
                float tmp = (ptr_point->depthConvergence * ptr_point->variance * 2 + ptr_parent->depthConvergence * ptr_parent->variance) / (ptr_point->variance * 2 + ptr_parent->variance);
                ptr_point->depthConvergence = ptr_point->depth = ptr_point->depthNew = tmp;
            }
        }
    }
    smoothDepth(nextLevel - 1);
}

void SingleTracker::probeDepthThisLevelDSO(int level, float lambda, Vector8f incrementPoseAffineThisLevel)
{
    const float maxPixelStep = 0.25;
    const float maxStepThreshold = 1e10;
    int num = 0;
    int count = -1;
    for(auto iter = refFrame->concernNormalPoints.at(level).begin(); iter != refFrame->concernNormalPoints.at(level).end(); iter++)
    {
        count++;
        num++;
        shared_ptr<Point> ptr_point = (*iter);
        if(ptr_point->isGood == false)
        {
            continue;
        }
        else
        {
            float incrementDepth = (-bAlpha2ml[count] - JAlpha_mul_JBeta2ml[count].dot(incrementPoseAffineThisLevel)) * JAlpha_mul_JAlphaSingle2ml[count] / (1 + lambda);
            // float incrementDepth = (-ptr_point->bAlpha - ptr_point->JAlphaT_mul_JBeta.dot(incrementPoseAffineThisLevel)) / ((ptr_point->JAlphaT_mul_JAlphaSingle + 1) * (1 + lambda));
            float maxStepThisPoint = maxPixelStep * ptr_point->maxstep;
            if(maxStepThisPoint > maxStepThreshold)
            {
                maxStepThisPoint = maxStepThreshold;
            }
            if(incrementDepth > maxStepThisPoint)
            {
                incrementDepth = maxStepThisPoint;
            }
            if(incrementDepth < -maxStepThisPoint)
            {
                incrementDepth = -maxStepThisPoint;
            }
            
            ptr_point->depthNew = ptr_point->depth + incrementDepth;
            if(ptr_point->depthNew < 1e-3)
            {
                ptr_point->depthNew = 1e-3;
            }
            if(ptr_point->depthNew > 50)
            {
                ptr_point->depthNew = 50;
            }
        }
        if(num == 60)
        {
            // cout << "bAlpha: " << ptr_point->bAlpha << ", JAlphaT_mul_JBeta: " << ptr_point->JAlphaT_mul_JBeta.matrix() << ", JAlphaT_mul_JAlphaSingle: " << (float)(1 / ptr_point->JAlphaT_mul_JAlphaSingle * (1+lambda)) << endl;
            cout << "\033[32m" << "depth sample: " << ptr_point->depthNew - ptr_point->depth << "\033[0m" << endl;
        }
    }
}

Vector2f SingleTracker::compareEnergyRegression(int level)
{
    if(!mutation)
    {
        return Vector2f(0, 0);
    }
    else
    {
        Vector2f energyRegression;
        for(auto iter = refFrame->concernNormalPoints.at(level).begin(); iter != refFrame->concernNormalPoints.at(level).end(); iter++)
        {
            shared_ptr<Point> ptr_point = (*iter);
            if(!ptr_point->isGoodNew)
            {
                continue;
            }
            else
            {
                energyRegression[0] += (ptr_point->depth - ptr_point->depthConvergence) * (ptr_point->depth - ptr_point->depthConvergence);
                energyRegression[1] += (ptr_point->depthNew - ptr_point->depthConvergence) * (ptr_point->depthNew - ptr_point->depthConvergence);
            }
        }
        return energyRegression;
    }
}


Vector3f SingleTracker::incrementalEquationLevelDSO(int level, const SE3& _poseRefToTarCurrent, AffineLight _affRefToTarCurrent)
{
    float fxL = fxG[level];
    float fyL = fyG[level];
    float cxL = cxG[level];
    float cyL = cyG[level];
    int wL = wG[level];
    int hL = hG[level];
    float totalErrorPoints = 0;
    Matrix3f R_mul_Kinv = (_poseRefToTarCurrent.rotationMatrix() * KinvG.at(level)).cast<float>();
    Vector3f t = _poseRefToTarCurrent.translation().cast<float>();

    HSchur.at(level).setZero();
    bSchur.at(level).setZero();
    HPoseAffine.at(level).setZero();
    bSchur.at(level).setZero();

    int num = 0;
    int totalNumPoint = refFrame->concernNormalPoints.at(level).size();
    float energyPoints = 0;
    float alphaEnergyPoints = alphaW * _poseRefToTarCurrent.translation().squaredNorm() * totalNumPoint;
    float alphaSmooth;
    if(alphaEnergyPoints > alphaK * totalNumPoint)
    {
        alphaSmooth = 0;
        alphaEnergyPoints = alphaK * totalNumPoint;
    }
    else
    {
        alphaSmooth = alphaW;
    }

    for(auto iter = refFrame->concernNormalPoints.at(level).begin(); iter != refFrame->concernNormalPoints.at(level).end(); iter++)
    {
        shared_ptr<Point> ptr_point = (*iter);
        ptr_point->maxstep = 1e10;
        if(!ptr_point->isGood)
        {
            energyPoints += ptr_point->energy[0];
            energyPoints += ptr_point->energy[1];
            ptr_point->energyNew = ptr_point->energy;
            ptr_point->isGoodNew = false;
            continue;
        }
        bool judgement = true;
        float energySinglePoint = 0;

        float componentJacobiAlphaSingle;
        Vector8f componentsJacobiBeta;
        float componentError;
        Vector8f JAlphaT_mul_JBeta;
        Matrix8f JBetaT_mul_JBeta;
        float JAlphaT_mul_JAlphaSingle;
        Vector8f bBeta;
        float bAlpha;

        componentJacobiAlphaSingle = 0;
        componentsJacobiBeta.setZero();
        componentError = 0;
        JAlphaT_mul_JBeta.setZero();
        JBetaT_mul_JBeta.setZero();
        JAlphaT_mul_JAlphaSingle = 0;
        bBeta.setZero();
        bAlpha = 0;
        
        
        for(int j = 0; j < patternNum; j++)
        {
            float refU, refV, refD;
            refU = ptr_point->positionX + pattern.at(j).first;
            refV = ptr_point->positionY + pattern.at(j).second;
            refD = ptr_point->depth;
            Vector3f P = R_mul_Kinv * Vector3f(refU, refV, 1) + t * refD;
            float tarU, tarV, tarD;
            tarU = P[0] / P[2];
            tarV = P[1] / P[2];
            tarD = refD / P[2];
            float K_mul_tarU, K_mul_tarV;
            K_mul_tarU = fxL * tarU + cxL;
            K_mul_tarV = fyL * tarV + cyL;

            
            if(K_mul_tarU <= 1 || K_mul_tarU >= wL - 1 || K_mul_tarV <= 1 || K_mul_tarV >= hL - 1)
            {
                judgement = false;
                break;
            }
            
            float refPixel = refFrame->interpolationPixel(level, refU, refV);
            Vector3f tarPixelDxDy = tarFrame->interpolationPixelDxDy(level, K_mul_tarU, K_mul_tarV);

            // if(num == 30)
            // {
            //     cout << tarU << " " << tarV << endl;
            //     cout << t.matrix() << endl;
            //     // cout << tarPixelDxDy.matrix() << endl;
            //     exit(1);
            // }
            
            if(!isfinite(refPixel) || !isfinite(tarPixelDxDy[0]))
            {
                judgement = false;
                break;
            }

            float residual = tarPixelDxDy[0] - _affRefToTarCurrent.a * refPixel - _affRefToTarCurrent.b;
            float hw = fabs(residual) < huberThreshold ? 1 : huberThreshold / fabs(residual);
            energySinglePoint += hw * residual * residual * (2 - hw);
            if(hw < 1)
            {
                hw = sqrtf(hw);
            }
            
            float tarDx_mul_fxL = hw * tarPixelDxDy[1] * fxL;
            float tarDy_mul_fyL = hw * tarPixelDxDy[2] * fyL;
            componentsJacobiBeta[0] = tarD * tarDx_mul_fxL;
            componentsJacobiBeta[1] = tarD * tarDy_mul_fyL;
            componentsJacobiBeta[2] = -tarD * (tarU * tarDx_mul_fxL + tarV * tarDy_mul_fyL);
            componentsJacobiBeta[3] = -tarU * tarV * tarDx_mul_fxL - (1 + tarV * tarV) * tarDy_mul_fyL;
            componentsJacobiBeta[4] = (1 + tarU * tarU) * tarDx_mul_fxL + tarU * tarV * tarDy_mul_fyL;
            componentsJacobiBeta[5] = -tarV * tarDx_mul_fxL + tarU * tarDy_mul_fyL;
            componentsJacobiBeta[6] = -hw * _affRefToTarCurrent.a * refPixel;
            componentsJacobiBeta[7] = -hw;
            componentJacobiAlphaSingle = tarDx_mul_fxL * (t[0] - t[2] * tarU) / P[2] + tarDy_mul_fyL * (t[1] - t[2] * tarV) / P[2];
            componentError = hw * residual;

            float maxstepSinglePoint = 1.0f / Vector2f((t[0] - t[2] * tarU) / P[2] * fxL, (t[1] - t[2] * tarV) / P[2] * fyL).norm();
            if(maxstepSinglePoint < ptr_point->maxstep)
            {
                ptr_point->maxstep = maxstepSinglePoint;
            }

            // if(num == 70 && j == 0)
            // {
            //     cout << "level: " << level << endl;
            //     cout << "threevec: " << tarPixelDxDy.matrix() << endl;
            //     cout << refU << ", " << refV << endl;
            //     cout << "tarD: " << tarD << endl;
            //     cout << "tarDy_mul_fyL: " <<tarDy_mul_fyL << endl;
            //     cout << componentsJacobiBeta.matrix() << endl;
            //     cout << componentError << endl;
            // }

            JAlphaT_mul_JBeta += componentJacobiAlphaSingle * componentsJacobiBeta;
            JBetaT_mul_JBeta += componentsJacobiBeta * componentsJacobiBeta.transpose();
            JAlphaT_mul_JAlphaSingle += componentJacobiAlphaSingle * componentJacobiAlphaSingle;
            bAlpha += componentJacobiAlphaSingle * componentError;
            bBeta += componentsJacobiBeta * componentError;
        }

        if(!judgement || energySinglePoint > ptr_point->outlierThreshold * 20)
        {
            energyPoints += ptr_point->energy[0];
            energyPoints += ptr_point->energy[1];
            ptr_point->status = 3;
            ptr_point->isGoodNew = false;
            ptr_point->energyNew = ptr_point->energy;
            continue;
        }
        else
        {
            energyPoints += energySinglePoint;
            ptr_point->isGoodNew = true;
            ptr_point->varianceNew = JAlphaT_mul_JAlphaSingle;
            ptr_point->energyNew[0] = energySinglePoint;
            ptr_point->energyNew[1] = (ptr_point->depthNew - 1) * (ptr_point->depthNew - 1);
            energyPoints += ptr_point->energyNew[1];

            bAlpha += alphaSmooth * (ptr_point->depthNew - 1);
            JAlphaT_mul_JAlphaSingle += alphaSmooth;
            if(alphaSmooth == 0)
            {
                bAlpha += couplingWeight * (ptr_point->depthNew - ptr_point->depthConvergence);
                JAlphaT_mul_JAlphaSingle += couplingWeight;
            }

            // if(t.norm() < 1)
            // {
            //     JAlphaT_mul_JAlphaSingle += 2 * 2 * totalNumPoint;
            // }
            // ptr_point->status = 1;
            // ptr_point->capability = errorPoint;

            HSchur.at(level) += JAlphaT_mul_JBeta * JAlphaT_mul_JBeta.transpose() / (JAlphaT_mul_JAlphaSingle + 1);
            HPoseAffine.at(level) += JBetaT_mul_JBeta;
            bSchur.at(level) += JAlphaT_mul_JBeta * bAlpha / (JAlphaT_mul_JAlphaSingle + 1);
            bPoseAffine.at(level) += bBeta;

            ptr_point->bAlphaNew = bAlpha;
            ptr_point->JAlphaT_mul_JAlphaSingleNew = JAlphaT_mul_JAlphaSingle;
            ptr_point->JAlphaT_mul_JBetaNew = JAlphaT_mul_JBeta;
            num++;
        }
    }
    // cout << "H: " << endl;
    // cout << HPoseAffine.at(level).matrix() << endl;
    // cout << "b: " << endl;
    // cout << bPoseAffine.at(level).matrix() << endl;
    // cout << "Hsch: " << endl;
    // cout << HSchur.at(level).matrix() << endl;
    // cout << "bsch: " << endl;
    // cout << bSchur.at(level).matrix() << endl;
    H.at(level) = HPoseAffine.at(level) - HSchur.at(level);
    b.at(level) = bPoseAffine.at(level) - bSchur.at(level);
    return Vector3f(energyPoints, alphaEnergyPoints, num);
}

Vector3f SingleTracker::incrementalEquationLevelFollowUpDSO(int level, const SE3& _poseRefToTarCurrent, AffineLight _affRefToTarCurrent)
{
    float fxL = fxG[level];
    float fyL = fyG[level];
    float cxL = cxG[level];
    float cyL = cyG[level];
    int wL = wG[level];
    int hL = hG[level];
    float totalErrorPoints = 0;
    Matrix3f R_mul_Kinv = (_poseRefToTarCurrent.rotationMatrix() * KinvG.at(level)).cast<float>();
    Vector3f t = _poseRefToTarCurrent.translation().cast<float>();

    HSchurNew.at(level).setZero();
    bSchurNew.at(level).setZero();
    HPoseAffineNew.at(level).setZero();
    bSchurNew.at(level).setZero();

    int num = 0;
    int totalNumPoint = refFrame->concernNormalPoints.at(level).size();
    float energyPoints = 0;
    float alphaEnergyPoints = alphaW * _poseRefToTarCurrent.translation().squaredNorm() * totalNumPoint;
    float alphaSmooth;
    if(alphaEnergyPoints > alphaK * totalNumPoint)
    {
        alphaSmooth = 0;
        alphaEnergyPoints = alphaK * totalNumPoint;
    }
    else
    {
        alphaSmooth = alphaW;
    }

    for(auto iter = refFrame->concernNormalPoints.at(level).begin(); iter != refFrame->concernNormalPoints.at(level).end(); iter++)
    {
        shared_ptr<Point> ptr_point = (*iter);
        ptr_point->maxstep = 1e10;
        if(!ptr_point->isGood)
        {
            energyPoints += ptr_point->energy[0];
            energyPoints += ptr_point->energy[1];
            ptr_point->energyNew = ptr_point->energy;
            ptr_point->isGoodNew = false;
            continue;
        }
        bool judgement = true;
        float energySinglePoint = 0;

        float componentJacobiAlphaSingle;
        Vector8f componentsJacobiBeta;
        float componentError;
        Vector8f JAlphaT_mul_JBeta;
        Matrix8f JBetaT_mul_JBeta;
        float JAlphaT_mul_JAlphaSingle;
        Vector8f bBeta;
        float bAlpha;

        componentJacobiAlphaSingle = 0;
        componentsJacobiBeta.setZero();
        componentError = 0;
        JAlphaT_mul_JBeta.setZero();
        JBetaT_mul_JBeta.setZero();
        JAlphaT_mul_JAlphaSingle = 0;
        bBeta.setZero();
        bAlpha = 0;
        
        
        for(int j = 0; j < patternNum; j++)
        {
            float refU, refV, refD;
            refU = ptr_point->positionX + pattern.at(j).first;
            refV = ptr_point->positionY + pattern.at(j).second;
            refD = ptr_point->depth;
            Vector3f P = R_mul_Kinv * Vector3f(refU, refV, 1) + t * refD;
            float tarU, tarV, tarD;
            tarU = P[0] / P[2];
            tarV = P[1] / P[2];
            tarD = refD / P[2];
            float K_mul_tarU, K_mul_tarV;
            K_mul_tarU = fxL * tarU + cxL;
            K_mul_tarV = fyL * tarV + cyL;

            
            if(K_mul_tarU <= 1 || K_mul_tarU >= wL - 1 || K_mul_tarV <= 1 || K_mul_tarV >= hL - 1)
            {
                judgement = false;
                break;
            }
            
            float refPixel = refFrame->interpolationPixel(level, refU, refV);
            Vector3f tarPixelDxDy = tarFrame->interpolationPixelDxDy(level, K_mul_tarU, K_mul_tarV);

            // if(num == 30)
            // {
            //     cout << tarU << " " << tarV << endl;
            //     cout << t.matrix() << endl;
            //     // cout << tarPixelDxDy.matrix() << endl;
            //     exit(1);
            // }
            
            if(!isfinite(refPixel) || !isfinite(tarPixelDxDy[0]))
            {
                judgement = false;
                break;
            }

            float residual = tarPixelDxDy[0] - _affRefToTarCurrent.a * refPixel - _affRefToTarCurrent.b;
            float hw = fabs(residual) < huberThreshold ? 1 : huberThreshold / fabs(residual);
            energySinglePoint += hw * residual * residual * (2 - hw);
            if(hw < 1)
            {
                hw = sqrtf(hw);
            }
            
            float tarDx_mul_fxL = hw * tarPixelDxDy[1] * fxL;
            float tarDy_mul_fyL = hw * tarPixelDxDy[2] * fyL;
            componentsJacobiBeta[0] = tarD * tarDx_mul_fxL;
            componentsJacobiBeta[1] = tarD * tarDy_mul_fyL;
            componentsJacobiBeta[2] = -tarD * (tarU * tarDx_mul_fxL + tarV * tarDy_mul_fyL);
            componentsJacobiBeta[3] = -tarU * tarV * tarDx_mul_fxL - (1 + tarV * tarV) * tarDy_mul_fyL;
            componentsJacobiBeta[4] = (1 + tarU * tarU) * tarDx_mul_fxL + tarU * tarV * tarDy_mul_fyL;
            componentsJacobiBeta[5] = -tarV * tarDx_mul_fxL + tarU * tarDy_mul_fyL;
            componentsJacobiBeta[6] = -hw * _affRefToTarCurrent.a * refPixel;
            componentsJacobiBeta[7] = -hw;
            componentJacobiAlphaSingle = tarDx_mul_fxL * (t[0] - t[2] * tarU) / P[2] + tarDy_mul_fyL * (t[1] - t[2] * tarV) / P[2];
            componentError = hw * residual;

            float maxstepSinglePoint = 1.0f / Vector2f((t[0] - t[2] * tarU) / P[2] * fxL, (t[1] - t[2] * tarV) / P[2] * fyL).norm();
            if(maxstepSinglePoint < ptr_point->maxstep)
            {
                ptr_point->maxstep = maxstepSinglePoint;
            }

            // if(num == 70 && j == 0)
            // {
            //     cout << "level: " << level << endl;
            //     cout << "threevec: " << tarPixelDxDy.matrix() << endl;
            //     cout << refU << ", " << refV << endl;
            //     cout << "tarD: " << tarD << endl;
            //     cout << "tarDy_mul_fyL: " <<tarDy_mul_fyL << endl;
            //     cout << componentsJacobiBeta.matrix() << endl;
            //     cout << componentError << endl;
            // }

            JAlphaT_mul_JBeta += componentJacobiAlphaSingle * componentsJacobiBeta;
            JBetaT_mul_JBeta += componentsJacobiBeta * componentsJacobiBeta.transpose();
            JAlphaT_mul_JAlphaSingle += componentJacobiAlphaSingle * componentJacobiAlphaSingle;
            bAlpha += componentJacobiAlphaSingle * componentError;
            bBeta += componentsJacobiBeta * componentError;
        }

        if(!judgement || energySinglePoint > ptr_point->outlierThreshold * 20)
        {
            energyPoints += ptr_point->energy[0];
            energyPoints += ptr_point->energy[1];
            ptr_point->status = 3;
            ptr_point->isGoodNew = false;
            ptr_point->energyNew = ptr_point->energy;
            continue;
        }
        else
        {
            energyPoints += energySinglePoint;
            ptr_point->varianceNew = JAlphaT_mul_JAlphaSingle;
            ptr_point->isGood = true;
            ptr_point->energyNew[0] = energySinglePoint;
            ptr_point->energyNew[1] = (ptr_point->depthNew - 1) * (ptr_point->depthNew - 1);
            energyPoints += ptr_point->energyNew[1];
            

            bAlpha += alphaSmooth * (ptr_point->depthNew - 1);
            JAlphaT_mul_JAlphaSingle += alphaSmooth;
            if(alphaSmooth == 0)
            {
                bAlpha += couplingWeight * (ptr_point->depthNew - ptr_point->depthConvergence);
                JAlphaT_mul_JAlphaSingle += couplingWeight;
            }

            // if(t.norm() < 1)
            // {
            //     JAlphaT_mul_JAlphaSingle += 2 * 2 * totalNumPoint;
            // }
            // ptr_point->status = 1;
            // ptr_point->capability = errorPoint;

            HSchurNew.at(level) += JAlphaT_mul_JBeta * JAlphaT_mul_JBeta.transpose() / (JAlphaT_mul_JAlphaSingle + 1);
            HPoseAffineNew.at(level) += JBetaT_mul_JBeta;
            bSchurNew.at(level) += JAlphaT_mul_JBeta * bAlpha / (JAlphaT_mul_JAlphaSingle + 1);
            bPoseAffineNew.at(level) += bBeta;

            ptr_point->bAlphaNew = bAlpha;
            ptr_point->JAlphaT_mul_JAlphaSingleNew = JAlphaT_mul_JAlphaSingle;
            ptr_point->JAlphaT_mul_JBetaNew = JAlphaT_mul_JBeta;
            num++;
        }
    }
    // cout << "H: " << endl;
    // cout << HPoseAffine.at(level).matrix() << endl;
    // cout << "b: " << endl;
    // cout << bPoseAffine.at(level).matrix() << endl;
    // cout << "Hsch: " << endl;
    // cout << HSchur.at(level).matrix() << endl;
    // cout << "bsch: " << endl;
    // cout << bSchur.at(level).matrix() << endl;
    H.at(level) = HPoseAffine.at(level) - HSchur.at(level);
    b.at(level) = bPoseAffine.at(level) - bSchur.at(level);
    return Vector3f(energyPoints, alphaEnergyPoints, num);
}


Vector3f SingleTracker::incrementalEquationDSO(int level, Matrix8f& Hsc, Vector8f& bsc, Matrix8f& HP, Vector8f& bP, const SE3& _poseRefToTarCurrent, AffineLight _affRefToTarCurrent)
{
    float fxL = fxG[level];
    float fyL = fyG[level];
    float cxL = cxG[level];
    float cyL = cyG[level];
    int wL = wG[level];
    int hL = hG[level];
    float totalErrorPoints = 0;
    Matrix3f R_mul_Kinv = (_poseRefToTarCurrent.rotationMatrix() * KinvG.at(level)).cast<float>();
    Vector3f t = _poseRefToTarCurrent.translation().cast<float>();

    Hsc.setZero();
    bsc.setZero();
    HP.setZero();
    bP.setZero();

    int totalNumPoint = refFrame->concernNormalPoints.at(level).size();
    float E = 0;
    int Enum = 0;


    int count = -1;
    for(auto iter = refFrame->concernNormalPoints.at(level).begin(); iter != refFrame->concernNormalPoints.at(level).end(); iter++)
    {
        count++;
        shared_ptr<Point> ptr_point = (*iter);
        ptr_point->maxstep = 1e10;
        if(!ptr_point->isGood)
        {
            E += ptr_point->energy[0];
            Enum++;
            ptr_point->energyNew = ptr_point->energy;
            ptr_point->isGoodNew = false;
            continue;
        }
        ptr_point->bAlphaNew = 0;
        ptr_point->JAlphaT_mul_JAlphaSingleNew = 0;
        ptr_point->JAlphaT_mul_JBetaNew.setZero();

        float componentJacobiAlphaSingle;
        Vector8f componentsJacobiBeta;
        float componentError;
        Vector8f JAlphaT_mul_JBeta;
        Matrix8f JBetaT_mul_JBeta;
        float JAlphaT_mul_JAlphaSingle;
        Vector8f bBeta;
        float bAlpha;

        componentJacobiAlphaSingle = 0;
        componentsJacobiBeta.setZero();
        componentError = 0;
        JAlphaT_mul_JBeta.setZero();
        JBetaT_mul_JBeta.setZero();
        JAlphaT_mul_JAlphaSingle = 0;
        bBeta.setZero();
        bAlpha = 0;

        JAlpha_mul_JAlphaSingle2mlNew[count] = 0;
        bAlpha2mlNew[count] = 0;
        JAlpha_mul_JBeta2mlNew[count].setZero();

        bool isGood = true;
        float energy = 0;
        for(int j = 0; j < patternNum; j++)
        {
            float refU, refV, refD;
            refU = ptr_point->positionX + pattern.at(j).first;
            refV = ptr_point->positionY + pattern.at(j).second;
            refD = ptr_point->depthNew;
            Vector3f P = R_mul_Kinv * Vector3f(refU, refV, 1) + t * refD;
            float tarU, tarV, tarD;
            tarU = P[0] / P[2];
            tarV = P[1] / P[2];
            tarD = refD / P[2];
            float K_mul_tarU, K_mul_tarV;
            K_mul_tarU = fxL * tarU + cxL;
            K_mul_tarV = fyL * tarV + cyL;
            if(K_mul_tarU <= 1 || K_mul_tarU >= wL - 1 || K_mul_tarV <= 1 || K_mul_tarV >= hL - 1)
            {
                isGood = false;
                break;
            }
            float refPixel = refFrame->interpolationPixel(level, refU, refV);
            Vector3f tarPixelDxDy = tarFrame->interpolationPixelDxDy(level, K_mul_tarU, K_mul_tarV);
            if(!isfinite(refPixel) || !isfinite(tarPixelDxDy[0]))
            {
                isGood = false;
                break;
            }
            float residual = tarPixelDxDy[0] - _affRefToTarCurrent.a * refPixel - _affRefToTarCurrent.b;
            float hw = fabs(residual) < huberThreshold ? 1 : huberThreshold / fabs(residual);
            energy += hw * residual * residual * (2 - hw);
            if(hw < 1)
            {
                hw = sqrtf(hw);
            }

            float tarDx_mul_fxL = hw * tarPixelDxDy[1] * fxL;
            float tarDy_mul_fyL = hw * tarPixelDxDy[2] * fyL;
            componentsJacobiBeta[0] = tarD * tarDx_mul_fxL;
            componentsJacobiBeta[1] = tarD * tarDy_mul_fyL;
            componentsJacobiBeta[2] = -tarD * (tarU * tarDx_mul_fxL + tarV * tarDy_mul_fyL);
            componentsJacobiBeta[3] = -tarU * tarV * tarDx_mul_fxL - (1 + tarV * tarV) * tarDy_mul_fyL;
            componentsJacobiBeta[4] = (1 + tarU * tarU) * tarDx_mul_fxL + tarU * tarV * tarDy_mul_fyL;
            componentsJacobiBeta[5] = -tarV * tarDx_mul_fxL + tarU * tarDy_mul_fyL;
            componentsJacobiBeta[6] = -hw * _affRefToTarCurrent.a * refPixel;
            componentsJacobiBeta[7] = -hw;
            componentJacobiAlphaSingle = tarDx_mul_fxL * (t[0] - t[2] * tarU) / P[2] + tarDy_mul_fyL * (t[1] - t[2] * tarV) / P[2];
            componentError = hw * residual;

            float maxstepSinglePoint = 1.0f / Vector2f((t[0] - t[2] * tarU) / P[2] * fxL, (t[1] - t[2] * tarV) / P[2] * fyL).norm();
            if(maxstepSinglePoint < ptr_point->maxstep)
            {
                ptr_point->maxstep = maxstepSinglePoint;
            }

            JAlpha_mul_JAlphaSingle2mlNew[count] += componentJacobiAlphaSingle * componentJacobiAlphaSingle;
            bAlpha2mlNew[count] += componentJacobiAlphaSingle * componentError;
            JAlpha_mul_JBeta2mlNew[count] += componentsJacobiBeta * componentJacobiAlphaSingle;

            JAlphaT_mul_JBeta += componentJacobiAlphaSingle * componentsJacobiBeta;
            JBetaT_mul_JBeta += componentsJacobiBeta * componentsJacobiBeta.transpose();
            JAlphaT_mul_JAlphaSingle += componentJacobiAlphaSingle * componentJacobiAlphaSingle;
            bAlpha += componentJacobiAlphaSingle * componentError;
            bBeta += componentsJacobiBeta * componentError;

        }
        if(!isGood || energy > ptr_point->outlierThreshold * 20)
        {
            E += ptr_point->energy[0];
            Enum++;
            ptr_point->isGoodNew = false;
            ptr_point->energyNew = ptr_point->energy;
            continue;
        }
        
        E += energy;
        Enum++;
        ptr_point->isGoodNew = true;
        ptr_point->energyNew[0] = energy;

        HP += JBetaT_mul_JBeta;
        bP += bBeta;
    }
    for(auto iter = refFrame->concernNormalPoints.at(level).begin(); iter != refFrame->concernNormalPoints.at(level).end(); iter++)
    {
        shared_ptr<Point> ptr_point = (*iter);
        if(!ptr_point->isGoodNew)
        {
            E += ptr_point->energy[1];
            Enum++;
        }
        else
        {
            ptr_point->energyNew[1] = (ptr_point->depthNew - 1) * (ptr_point->depthNew - 1);
            E += ptr_point->energyNew[1];
            Enum++;
        }
    }
    float alphaEnergyPoints = alphaW * _poseRefToTarCurrent.translation().squaredNorm() * totalNumPoint;
    float alphaSmooth;
    if(alphaEnergyPoints > alphaK * totalNumPoint)
    {
        alphaSmooth = 0;
        alphaEnergyPoints = alphaK * totalNumPoint;
    }
    else
    {
        alphaSmooth = alphaW;
    }

    count = -1;
    for(auto iter = refFrame->concernNormalPoints.at(level).begin(); iter != refFrame->concernNormalPoints.at(level).end(); iter++)
    {
        count++;
        shared_ptr<Point> ptr_point = (*iter);
        if(!ptr_point->isGoodNew)
        {
            continue;
        }
        ptr_point->varianceNew = JAlpha_mul_JAlphaSingle2mlNew[count];
        ptr_point->JAlphaT_mul_JAlphaSingleNew = JAlpha_mul_JAlphaSingle2mlNew[count];
        
        bAlpha2mlNew[count] += alphaSmooth * (ptr_point->depthNew - 1);
        JAlpha_mul_JAlphaSingle2mlNew[count] += alphaSmooth;

        if(alphaSmooth == 0)
        {
            bAlpha2mlNew[count] += couplingWeight * (ptr_point->depthNew - ptr_point->depthConvergence);
            JAlpha_mul_JAlphaSingle2mlNew[count] += couplingWeight;
        }

        JAlpha_mul_JAlphaSingle2mlNew[count] = 1 / (1 + JAlpha_mul_JAlphaSingle2mlNew[count]);
        Hsc += JAlpha_mul_JBeta2mlNew[count] * JAlpha_mul_JBeta2mlNew[count].transpose() * JAlpha_mul_JAlphaSingle2mlNew[count];
        bsc += JAlpha_mul_JBeta2mlNew[count] * bAlpha2mlNew[count] * JAlpha_mul_JAlphaSingle2mlNew[count];
    }
    HP(0, 0) += alphaSmooth * totalNumPoint;
    HP(1, 1) += alphaSmooth * totalNumPoint;
    HP(2, 2) += alphaSmooth * totalNumPoint;
    Vector3f tlog = _poseRefToTarCurrent.log().head<3>().cast<float>();
    bP[0] += tlog[0] * alphaSmooth * totalNumPoint;
    bP[1] += tlog[1] * alphaSmooth * totalNumPoint;
    bP[2] += tlog[2] * alphaSmooth * totalNumPoint;

    return Vector3f(E, alphaEnergyPoints, Enum);
}