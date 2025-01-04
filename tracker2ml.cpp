#include "tracker.h"
void SingleTracker::optimizeRelativePose2ml()
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
            }
        }
    }

    Vector3f lastResult = Vector3f::Zero();
    for(int i = LEVEL - 1; i >= 0; i--)
    {
        if(i != LEVEL - 1)
        {
            spreadDepthToNextLevel2ml(i + 1);
        }
        Matrix8f HThisLevel;
        Vector8f bThisLevel;
        int iterations = 0;
        int failedAttempt = 0;
        float lambda = 0.1;
        int numPointThisLevel = refFrame->concernNormalPoints.at(i).size();
        resetPoints(i);

        while(true)
        {
            Vector3f energyOld = incrementalEquation2ml(i);
            for(int j = 0; j < 8; j++)
            {
                HThisLevel(j, j) *= (1 + lambda);
            }
            HThisLevel = HPoseAffine.at(i) - HSchur.at(i) * (1 / (1 + lambda));
            bThisLevel = bPoseAffine.at(i) - bSchur.at(i) * (1 / (1 + lambda));
            HThisLevel = weightEquation * HThisLevel * weightEquation * (0.01 / (wG[i] * hG[i]));
            bThisLevel = weightEquation * bThisLevel * (0.01 / (wG[i] * hG[i]));

            Vector8f incrementPoseAndAffine;
            incrementPoseAndAffine.head<6>() = -(weightEquation.toDenseMatrix().topLeftCorner<6, 6>() * HThisLevel.topLeftCorner<6, 6>().ldlt().solve(bThisLevel.head<6>()));
            incrementPoseAndAffine.tail<2>().setZero();

            relativaPoseNew = SE3::exp(incrementPoseAndAffine.head<6>().cast<double>()) * relativaPose;
            relativeAffineNew = relativeAffine;
            probeDepth2ml(i, lambda, incrementPoseAndAffine);

            Vector3f energyNew = checkEnergy2ml(i);
            Vector2f energyReg = compareEnergyRegression2ml(i);

            float energyTotallyNew = energyNew[0] + energyNew[1] + energyReg[1];
            float energyTotallyCurrent = energyOld[0] + energyOld[1] + energyReg[0]; 

            bool approval = energyTotallyNew < energyTotallyCurrent;
            cout << "old: " << energyTotallyCurrent << ", new: " << energyTotallyNew << endl;

            if(approval)
            {
                if(energyNew[1] == alphaK * numPointThisLevel)
                {
                    mutation = true;
                }
                energyOld = energyNew;
                relativaPose = relativaPoseNew;
                relativeAffine = relativeAffineNew;
                updateDepthThisLevel2ml(i);
                smoothDepth(i);
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
            lastResult = energyOld;
            if(!(incrementPoseAndAffine.norm() > 1e-4) || iterations > maxIterations[i] || failedAttempt >= 2)
            {
                break;
            }
            iterations++;
        }
        exit(1);
    }
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
        climbDepthLevel2ml(level);
    }
}

Vector2f SingleTracker::compareEnergyRegression2ml(int level)
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
                energyRegression[0] += ptr_point->depth - ptr_point->depthConvergence;
                energyRegression[1] += ptr_point->depthNew - ptr_point->depthConvergence;
            }
        }
        return energyRegression;
    }
}


void SingleTracker::climbDepthLevel2ml(int level)
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
            shared_ptr<Point> ptr_parent = refFrame->concernNormalPoints.at(level).at(ptr_point->indexParent);
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

void SingleTracker::updateDepthThisLevel2ml(int level)
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
}

Vector3f SingleTracker::checkEnergy2ml(int level)
{
    float fxL = fxG[level];
    float fyL = fyG[level];
    float cxL = cxG[level];
    float cyL = cyG[level];
    int wL = wG[level];
    int hL = hG[level];
    float totalErrorPoints = 0;
    Matrix3f R_mul_Kinv = (relativaPoseNew.rotationMatrix() * KinvG.at(level)).cast<float>();
    Vector3f t = relativaPoseNew.translation().cast<float>();

    int num = 0;
    int totalNumPoint = refFrame->concernNormalPoints.at(level).size();
    float energyPoints = 0;
    float alphaEnergyPoints = alphaW * relativaPoseNew.translation().squaredNorm() * totalNumPoint;
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
        if(!ptr_point->isGood)
        {
            energyPoints += ptr_point->energy[0];
            energyPoints += ptr_point->energy[1];
            continue;
        }

        bool judgement = true;
        float energySinglePoint = 0;

        // float componentJacobiAlphaSingle;
        // Vector8f componentsJacobiBeta;
        // float componentError;
        // Vector8f JAlphaT_mul_JBeta;
        // Matrix8f JBetaT_mul_JBeta;
        // float JAlphaT_mul_JAlphaSingle;
        // Vector8f bBeta;
        // float bAlpha;

        // componentJacobiAlphaSingle = 0;
        // componentsJacobiBeta.setZero();
        // componentError = 0;
        // JAlphaT_mul_JBeta.setZero();
        // JBetaT_mul_JBeta.setZero();
        // JAlphaT_mul_JAlphaSingle = 0;
        // bBeta.setZero();
        // bAlpha = 0;
        
        
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
            
            if(!isfinite(refPixel) || !isfinite(tarPixelDxDy[0]))
            {
                judgement = false;
                break;
            }

            float residual = tarPixelDxDy[0] - relativeAffineNew[0] * refPixel - relativeAffineNew[1];
            float hw = fabs(residual) < huberThreshold ? 1 : huberThreshold / fabs(residual);
            energySinglePoint += hw * residual * residual * (2 - hw);
            // if(hw < 1)
            // {
            //     hw = sqrtf(hw);
            // }
            
            // float tarDx_mul_fxL = hw * tarPixelDxDy[1] * fxL;
            // float tarDy_mul_fyL = hw * tarPixelDxDy[2] * fyL;
            // componentsJacobiBeta[0] = tarD * tarDx_mul_fxL;
            // componentsJacobiBeta[1] = tarD * tarDy_mul_fyL;
            // componentsJacobiBeta[2] = -tarD * (tarU * tarDx_mul_fxL + tarV * tarDy_mul_fyL);
            // componentsJacobiBeta[3] = -tarU * tarV * tarDx_mul_fxL - (1 + tarV * tarV) * tarDy_mul_fyL;
            // componentsJacobiBeta[4] = (1 + tarU * tarU) * tarDx_mul_fxL + tarU * tarV * tarDy_mul_fyL;
            // componentsJacobiBeta[5] = -tarV * tarDx_mul_fxL + tarU * tarDy_mul_fyL;
            // componentsJacobiBeta[6] = -hw * relativeAffineNew[0] * refPixel;
            // componentsJacobiBeta[7] = -hw;
            // componentJacobiAlphaSingle = tarDx_mul_fxL * (t[0] - t[2] * tarU) / P[2] + tarDy_mul_fyL * (t[1] - t[2] * tarV) / P[2];
            // componentError = hw * residual;

            // float maxstepSinglePoint = 1.0f / Vector2f((t[0] - t[2] * tarU) / P[2] * fxL, (t[1] - t[2] * tarV) / P[2] * fyL).norm();
            // if(maxstepSinglePoint < ptr_point->maxstep)
            // {
            //     ptr_point->maxstep = maxstepSinglePoint;
            // }

            // JAlphaT_mul_JBeta += componentJacobiAlphaSingle * componentsJacobiBeta;
            // JBetaT_mul_JBeta += componentsJacobiBeta * componentsJacobiBeta.transpose();
            // JAlphaT_mul_JAlphaSingle += componentJacobiAlphaSingle * componentJacobiAlphaSingle;
            // bAlpha += componentJacobiAlphaSingle * componentError;
            // bBeta += componentsJacobiBeta * componentError;
        }

        if(!judgement || energySinglePoint > ptr_point->outlierThreshold * 20)
        {
            energyPoints += ptr_point->energy[0];
            energyPoints += ptr_point->energy[1];
            // ptr_point->isGoodNew = false;
            // ptr_point->energyNew = ptr_point->energy;
            continue;
        }

        energyPoints += energySinglePoint;
        // ptr_point->varianceNew = JAlphaT_mul_JAlphaSingle;
        // ptr_point->isGoodNew = true;
        // ptr_point->energyNew[0] = energySinglePoint;
        // ptr_point->energyNew[1] = (ptr_point->depth - 1) * (ptr_point->depth - 1);
        energyPoints += (ptr_point->depthNew - 1) * (ptr_point->depthNew - 1);

        // bAlpha += alphaSmooth * (ptr_point->depthNew - 1);
        // JAlphaT_mul_JAlphaSingle += alphaSmooth;
        // if(alphaSmooth == 0)
        // {
        //     bAlpha += couplingWeight * (ptr_point->depthNew - ptr_point->depthConvergence);
        //     JAlphaT_mul_JAlphaSingle += couplingWeight;
        // }

        // HSchur.at(level) += JAlphaT_mul_JBeta * JAlphaT_mul_JBeta.transpose() / (JAlphaT_mul_JAlphaSingle + 1);
        // HPoseAffine.at(level) += JBetaT_mul_JBeta;
        // bSchur.at(level) += JAlphaT_mul_JBeta * bAlpha / (JAlphaT_mul_JAlphaSingle + 1);
        // bPoseAffine.at(level) += bBeta;

        // ptr_point->bAlpha = bAlpha;
        // ptr_point->JAlphaT_mul_JAlphaSingle = componentJacobiAlphaSingle;
        // ptr_point->JAlphaT_mul_JBeta = JAlphaT_mul_JBeta;
        num++;
    }
    // H.at(level) = HPoseAffine.at(level) - HSchur.at(level);
    // b.at(level) = bPoseAffine.at(level) - bSchur.at(level);
    return Vector3f(energyPoints, alphaEnergyPoints, num);
}


void SingleTracker::probeDepth2ml(int level, float lambda, Vector8f incrementPoseAffineThisLevel)
{
    const float maxPixelStep = 0.25;
    const float maxStepThreshold = 1e10;
    int num = 0;
    for(auto iter = refFrame->concernNormalPoints.at(level).begin(); iter != refFrame->concernNormalPoints.at(level).end(); iter++)
    {
        num++;
        shared_ptr<Point> ptr_point = (*iter);
        if(!ptr_point->isGoodNew)
        {
            continue;
        }
        else
        {
            float incrementDepth = (-ptr_point->bAlpha - ptr_point->JAlphaT_mul_JBeta.dot(incrementPoseAffineThisLevel)) / (ptr_point->JAlphaT_mul_JAlphaSingle + 1) / (1 + lambda);
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
        // if(num == 60)
        // {
        //     // cout << "bAlpha: " << ptr_point->bAlpha << ", JAlphaT_mul_JBeta: " << ptr_point->JAlphaT_mul_JBeta.matrix() << ", JAlphaT_mul_JAlphaSingle: " << (float)(1 / ptr_point->JAlphaT_mul_JAlphaSingle * (1+lambda)) << endl;
        //     cout << "\033[32m" << "depth sample: " << ptr_point->depthNew - ptr_point->depth << "\033[0m" << endl;
        // }
    }
}

void SingleTracker::spreadDepthToNextLevel2ml(int nextLevel)
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

Vector3f SingleTracker::incrementalEquation2ml(int level)
{
    float fxL = fxG[level];
    float fyL = fyG[level];
    float cxL = cxG[level];
    float cyL = cyG[level];
    int wL = wG[level];
    int hL = hG[level];
    float totalErrorPoints = 0;
    Matrix3f R_mul_Kinv = (relativaPose.rotationMatrix() * KinvG.at(level)).cast<float>();
    Vector3f t = relativaPose.translation().cast<float>();
    HPoseAffine.at(level).setZero();
    bPoseAffine.at(level).setZero();
    HSchur.at(level).setZero();
    bSchur.at(level).setZero();

    int num = 0;
    int totalNumPoints = refFrame->concernNormalPoints.at(level).size();
    float energyPoints = 0;

    float alphaEnergyPoints = alphaW * relativaPose.translation().squaredNorm() * totalNumPoints;
    float alphaSmooth;
    if(alphaEnergyPoints > alphaK * totalNumPoints)
    {
        alphaSmooth = 0;
        alphaEnergyPoints = alphaK * totalNumPoints;
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
            ptr_point->isGoodNew = ptr_point->isGood;
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

            float residual = tarPixelDxDy[0] - relativeAffine[0] * refPixel - relativeAffine[1];
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
            componentsJacobiBeta[6] = -hw * relativeAffine[0] * refPixel;
            componentsJacobiBeta[7] = -hw;
            componentJacobiAlphaSingle = tarDx_mul_fxL * (t[0] - t[2] * tarU) / P[2] + tarDy_mul_fyL * (t[1] - t[2] * tarV) / P[2];
            componentError = hw * residual;

            float maxstepSinglePoint = 1.0f / Vector2f((t[0] - t[2] * tarU) / P[2] * fxL, (t[1] - t[2] * tarV) / P[2] * fyL).norm();
            if(maxstepSinglePoint < ptr_point->maxstep)
            {
                ptr_point->maxstep = maxstepSinglePoint;
            }

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
            ptr_point->isGoodNew = false;
            ptr_point->energyNew = ptr_point->energy;
            continue;
        }

        energyPoints += energySinglePoint;
        ptr_point->varianceNew = JAlphaT_mul_JAlphaSingle;
        ptr_point->isGoodNew = true;
        ptr_point->energyNew[0] = energySinglePoint;
        ptr_point->energyNew[1] = (ptr_point->depth - 1) * (ptr_point->depth - 1);
        energyPoints += ptr_point->energyNew[1];

        bAlpha += alphaSmooth * (ptr_point->depth - 1);
        JAlphaT_mul_JAlphaSingle += alphaSmooth;
        if(alphaSmooth == 0)
        {
            bAlpha += couplingWeight * (ptr_point->depth - ptr_point->depthConvergence);
            JAlphaT_mul_JAlphaSingle += couplingWeight;
        }

        HSchur.at(level) += JAlphaT_mul_JBeta * JAlphaT_mul_JBeta.transpose() / (JAlphaT_mul_JAlphaSingle + 1);
        HPoseAffine.at(level) += JBetaT_mul_JBeta;
        bSchur.at(level) += JAlphaT_mul_JBeta * bAlpha / (JAlphaT_mul_JAlphaSingle + 1);
        bPoseAffine.at(level) += bBeta;

        ptr_point->bAlpha = bAlpha;
        ptr_point->JAlphaT_mul_JAlphaSingle = componentJacobiAlphaSingle;
        ptr_point->JAlphaT_mul_JBeta = JAlphaT_mul_JBeta;
        num++;
    }
    H.at(level) = HPoseAffine.at(level) - HSchur.at(level);
    b.at(level) = bPoseAffine.at(level) - bSchur.at(level);
    return Vector3f(energyPoints, alphaEnergyPoints, num);
}