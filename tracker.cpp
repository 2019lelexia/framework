#include "tracker.h"

SingleTracker::SingleTracker() : relativaPose(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero())
{
    relativaPoseNew = relativaPose;
    relativeAffineNew = relativeAffineNew;
    pattern.emplace_back(0, 0);
    pattern.emplace_back(2, 0);
    pattern.emplace_back(-2, 0);
    pattern.emplace_back(0, 2);
    pattern.emplace_back(0, -2);
    pattern.emplace_back(1, 1);
    pattern.emplace_back(-1, 1);
    pattern.emplace_back(-1, -1);
    patternNum = 8;
    huberThreshold = 9;
    initializationDropThreshold = 5;
    for(int i = 0; i < LEVEL; i++)
    {
        H.push_back(Matrix8f::Zero());
        b.push_back(Vector8f::Zero());
        HSchur.push_back(Matrix8f::Zero());
        bSchur.push_back(Vector8f::Zero());
        HPoseAffine.push_back(Matrix8f::Zero());
        bPoseAffine.push_back(Vector8f::Zero());
        HSchurNew.push_back(Matrix8f::Zero());
        bSchurNew.push_back(Vector8f::Zero());
        HPoseAffineNew.push_back(Matrix8f::Zero());
        bPoseAffineNew.push_back(Vector8f::Zero());
        numPoints.push_back(0);
    }
    weightEquation.diagonal()[0] = weightEquation.diagonal()[1] = weightEquation.diagonal()[2] = 1.0;
    weightEquation.diagonal()[3] = weightEquation.diagonal()[4] = weightEquation.diagonal()[5] = 0.5;
    weightEquation.diagonal()[6] = 10.0;
    weightEquation.diagonal()[7] = 1000.0;


    debugSign = 0;
}

SingleTracker::~SingleTracker()
{}

void SingleTracker::setRefFrame(shared_ptr<Frame> _refFrame)
{
    refFrame = _refFrame;
    mutation = false;
    positionMutation = 0;
    maxIterations.push_back(5);
    maxIterations.push_back(5);
    maxIterations.push_back(10);
    maxIterations.push_back(20);
}

void SingleTracker::setTarFrame(shared_ptr<Frame> _tarFrame)
{
    tarFrame = _tarFrame;
    relativeAffine = refFrame->affine.deviationAffineLight(refFrame->exposure, tarFrame->exposure, refFrame->affine, tarFrame->affine);
}

void SingleTracker::setTrajectory(shared_ptr<Trajectory> _trajectoryer)
{
    trajectory = _trajectoryer;
}

void SingleTracker::optimizeRelativePose()
{
    // refFrame->makePointCloudFlann();
    // cout << "check point cloud flann ok" << endl;

    // auto all = refFrame->concernNormalPoints.at(0);
    // auto tmp = refFrame->concernNormalPoints.at(0).at(101);
    // cout << "position: " << tmp->getPositionX() << ", " << tmp->getPositionY() << endl;
    // for(int i = 0; i < 5; i++)
    // {
    //     auto t = all.at(tmp->indexNeighbour[i]);
    //     cout << "neighbour" << i <<": " << t->getPositionX() << ", " << t->getPositionY() << endl;
    // }
    // auto up = refFrame->concernNormalPoints.at(1);
    // auto t = up.at(tmp->indexParent);
    // cout << "parent: " << t->getPositionX() << ", " << t->getPositionY() << endl;
    // exit(1);
    for(int i = LEVEL - 1; i >= 0; i--)
    {
        if(i != LEVEL - 1)
        {
            spreadDepthToNextLevel(i);
        }
        int iterations = 0;
        float lambda = 0.1;
        Matrix8f HThisLevel;
        Vector8f bThisLevel;
        float formerError = 0;
        float latterError = 0;
        int numLostPoints = 0;
        bool acceptance = true;
        int failedAttempt = 0;
        while(true)
        {
            if(acceptance)
            {
                // cout << "p1" << endl;
                Vector2f formerResults;
                formerResults = incrementalEquationLevel(i);
                formerError = formerResults[0];
                // cout << "p2" << endl;
                cout << "formerError: " << formerError << endl;
            }
            HThisLevel = HPoseAffine.at(i);
            // cout << HPoseAffine.at(i).matrix() << endl;
            for(int j = 0; j < 8; j++)
            {
                HThisLevel(j, j) *= (1 + lambda);
            }
            // cout << HSchur.at(i).matrix() << endl;
            HThisLevel -= HSchur.at(i) * (1 / (1 + lambda));
            bThisLevel = bPoseAffine.at(i) - bSchur.at(i) * (1 / (1 + lambda));
            HThisLevel = weightEquation * HThisLevel * weightEquation * (0.01 / (wG[i] * hG[i]));
            bThisLevel = weightEquation * bThisLevel * (0.01 / (wG[i] * hG[i]));

            Vector8f incrementPoseAffine;
            // incrementPoseAffine = -(weightEquation * (HThisLevel.ldlt().solve(bThisLevel)));
            incrementPoseAffine.head<6>() = -(weightEquation.toDenseMatrix().topLeftCorner<6, 6>() * HThisLevel.topLeftCorner<6, 6>().ldlt().solve(bThisLevel.head<6>()));
            incrementPoseAffine.tail<2>().setZero();
            // cout << "look: " << endl;
            // cout << HThisLevel.matrix() << endl;
            // cout << bThisLevel.matrix() << endl;
            cout << "increment: " << endl;
            cout << incrementPoseAffine.matrix() << endl;
            // exit(0);
            relativaPoseNew = SE3::exp(incrementPoseAffine.head<6>().cast<double>()) * relativaPose;
            relativeAffineNew = relativeAffine;
            // relativeAffineNew[0] += incrementPoseAffine[6];
            // relativeAffineNew[1] += incrementPoseAffine[7];
            probeDepthThisLevel(i, lambda, incrementPoseAffine);
            
            Vector2f latterResults = checkUpdateLevel(i);
            latterError = latterResults[0];
            numLostPoints = latterResults[1];
            cout << "latterError: " << latterError << endl;
            iterations++;
            acceptance = true;
            if(latterError > formerError || numLostPoints > (int)(numPoints.at(i) * 0.1))
            {
                acceptance = false;
            }
            if(acceptance)
            {
                cout << "acceptance" << endl;
                failedAttempt = 0;
                relativaPose = relativaPoseNew;
                relativeAffine = relativeAffineNew;
                updateDepthThisLevel(i);
                formerError = latterError;
            }
            else
            {
                failedAttempt += 1;
                lambda *= 10;
            }

            if(lambda > 10000 || failedAttempt >= 2 || iterations > 5)
            {
                cout << "\033[31m" << "the times of attempts is too much or lambda is too large" << "\033[0m" << endl;
                break;
            }
        }
    }
    cout << relativaPose.matrix() << endl;
    // refFrame->visualizePointCloudLevel(0);
}



void SingleTracker::probeDepthThisLevel(int level, float lambda, Vector8f incrementPoseAffineThisLevel)
{
    int num = 0;
    for(auto iter = refFrame->concernNormalPoints.at(level).begin(); iter != refFrame->concernNormalPoints.at(level).end(); iter++)
    {
        num++;
        shared_ptr<Point> ptr_point = (*iter);
        if(ptr_point->status != 1)
        {
            continue;
        }
        else
        {
            float incrementDepth = (-ptr_point->bAlpha - ptr_point->JAlphaT_mul_JBeta.transpose() * incrementPoseAffineThisLevel) / ((ptr_point->JAlphaT_mul_JAlphaSingle + 1) * (1 + lambda));
            ptr_point->depthNew = ptr_point->depth + incrementDepth;
            if(ptr_point->depthNew < 1e-2)
            {
                ptr_point->depthNew = 1e-2;
            }
            if(ptr_point->depthNew > 10)
            {
                ptr_point->depthNew = 10;
            }
        }
        if(num == 100)
        {
            cout << "bAlpha: " << ptr_point->bAlpha << ", JAlphaT_mul_JBeta: " << ptr_point->JAlphaT_mul_JBeta.matrix() << ", JAlphaT_mul_JAlphaSingle: " << (float)(1 / ptr_point->JAlphaT_mul_JAlphaSingle) << endl;
            cout << "\033[32m" << "depth sample: " << ptr_point->depthNew - ptr_point->depth << "\033[0m" << endl;
        }
    }
}

void SingleTracker::updateDepthThisLevel(int level)
{
    for(auto iter = refFrame->concernNormalPoints.at(level).begin(); iter != refFrame->concernNormalPoints.at(level).end(); iter++)
    {
        shared_ptr<Point> ptr_point = (*iter);
        if(ptr_point->status != 1)
        {
            continue;
        }
        else
        {
            ptr_point->depth = ptr_point->depthNew;
        }
        if(debugSign == 1)
        {
            cout << ptr_point->depthNew << endl;
        }
    }
    if(debugSign)
    {
        exit(1);
    }
}

void SingleTracker::spreadDepthToNextLevel(int nextLevel)
{
    vector<shared_ptr<Point>> thisLevelPoints = refFrame->concernNormalPoints.at(nextLevel + 1);
    for(auto iter = refFrame->concernNormalPoints.at(nextLevel).begin(); iter != refFrame->concernNormalPoints.at(nextLevel).end(); iter++)
    {
        shared_ptr<Point> ptr_point = (*iter);
        shared_ptr<Point> ptr_point_this_level = thisLevelPoints.at(ptr_point->indexParent);
        if(ptr_point_this_level->status != 3)
        {
            ptr_point->depth = ptr_point_this_level->depth;
        }
        else
        {
            ptr_point->depth = 1;
        }
    }
}

void SingleTracker::exchangeOriginalAndNew()
{
    relativaPose = relativaPoseNew;
    relativeAffine = relativeAffineNew;
}

void SingleTracker::resetPoints(int level)
{
    for(auto iter = refFrame->concernNormalPoints.at(level).begin(); iter != refFrame->concernNormalPoints.at(level).end(); iter++)
    {
        shared_ptr<Point> ptr_point = (*iter);
        ptr_point->energy.setZero();
        ptr_point->depthNew = ptr_point->depth;
    }
}

Vector2f SingleTracker::incrementalEquationLevel(int level)
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

    HSchur.at(level).setZero();
    bSchur.at(level).setZero();
    HPoseAffine.at(level).setZero();
    bSchur.at(level).setZero();

    int num = 0;
    int totalNumPoint = refFrame->concernNormalPoints.at(level).size();
    

    for(auto iter = refFrame->concernNormalPoints.at(level).begin(); iter != refFrame->concernNormalPoints.at(level).end(); iter++)
    {
        shared_ptr<Point> ptr_point = (*iter);
        float errorPoint = 0;

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
        
        bool judgement = true;

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
            if(hw < 1)
            {
                hw = sqrtf(hw);
            }
            errorPoint += hw * residual * residual * (2 - hw);
            
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
        if(!judgement)
        {
            ptr_point->status = 3;
            continue;
        }
        else
        {
            if(t.norm() < 1)
            {
                JAlphaT_mul_JAlphaSingle += 2 * 2 * totalNumPoint;
            }
            ptr_point->status = 1;
            ptr_point->capability = errorPoint;
            ptr_point->bAlpha = bAlpha;
            ptr_point->JAlphaT_mul_JAlphaSingle = JAlphaT_mul_JAlphaSingle;
            ptr_point->JAlphaT_mul_JBeta = JAlphaT_mul_JBeta;
            totalErrorPoints += errorPoint;

            HSchur.at(level) += JAlphaT_mul_JBeta * JAlphaT_mul_JBeta.transpose() / (JAlphaT_mul_JAlphaSingle + 1);
            HPoseAffine.at(level) += JBetaT_mul_JBeta;
            bSchur.at(level) += JAlphaT_mul_JBeta * bAlpha / (JAlphaT_mul_JAlphaSingle + 1);
            bPoseAffine.at(level) += bBeta;
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
    return Vector2f(totalErrorPoints, num);
}

Vector2f SingleTracker::checkUpdateLevel(int level)
{
    float fxL = fxG[level];
    float fyL = fyG[level];
    float cxL = cxG[level];
    float cyL = cyG[level];
    int wL = wG[level];
    int hL = hG[level];
    float totalErrorPoints = 0;
    int totalOutOfBoundaryPoints = 0;
    Matrix3f R_mul_Kinv = (relativaPoseNew.rotationMatrix() * KinvG.at(level)).cast<float>();
    Vector3f t = relativaPoseNew.translation().cast<float>();
    // cout << t.matrix() << endl;

    for(auto iter = refFrame->concernNormalPoints.at(level).begin(); iter != refFrame->concernNormalPoints.at(level).end(); iter++)
    {
        shared_ptr<Point> ptr_point = (*iter);
        float errorPoint;
        errorPoint = 0;
        bool judgement = true;
        if(ptr_point->status == 3)
        {
            continue;
        }

        for(int j = 0; j < patternNum; j++)
        {
            float refU, refV, refD;
            refU = ptr_point->positionX + pattern.at(j).first;
            refV = ptr_point->positionY + pattern.at(j).second;
            refD = ptr_point->depthNew;
            Vector3f P = R_mul_Kinv * Vector3f(refU, refV, 1) + t * refD;
            // cout << "refD: " << refD << endl;
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
            // cout << "okok" << endl;
            // cout << refU << " , " << refV << endl;
            // cout << tarU << " , " << tarV << endl;
            float refPixel = refFrame->interpolationPixel(level, refU, refV);
            Vector3f tarPixelDxDy = tarFrame->interpolationPixelDxDy(level, K_mul_tarU, K_mul_tarV);
            if(!isfinite(refPixel) || !isfinite(tarPixelDxDy[0]))
            {
                judgement = false;
                break;
            }

            float residual = tarPixelDxDy[0] - relativeAffineNew[0] * refPixel - relativeAffineNew[1];
            float hw = fabs(residual) < huberThreshold ? 1 : huberThreshold / fabs(residual);
            if(hw < 1)
            {
                hw = sqrtf(hw);
            }
            errorPoint += hw * residual * residual * (2 - hw);
        }
        if(!judgement)
        {
            totalOutOfBoundaryPoints += 1;
            continue;
        }
        else
        {
            totalErrorPoints += errorPoint;
        }
    }
    return  Vector2f(totalErrorPoints, totalOutOfBoundaryPoints);
}

template<typename T>
float findMedianOptimized(vector<T>& data) {
    size_t size = data.size();
    size_t mid = size / 2;
    
    if (size % 2 != 0)
    {
        nth_element(data.begin(), data.begin() + mid, data.end());
        return data[mid];
    }
    else
    {
        nth_element(data.begin(), data.begin() + mid - 1, data.end());
        int left = data[mid - 1];
        nth_element(data.begin(), data.begin() + mid, data.end());
        int right = data[mid];
        return (left + right) / 2.0f;
    }
}

void SingleTracker::smoothDepth(int level)
{
    if(mutation == false)
    {
        for(auto iter = refFrame->concernNormalPoints.at(level).begin(); iter != refFrame->concernNormalPoints.at(level).end(); iter++)
        {
            shared_ptr<Point> ptr_point = (*iter);
            ptr_point->depthConvergence = 1;
        }
        return;
    }
    for(auto iter = refFrame->concernNormalPoints.at(level).begin(); iter != refFrame->concernNormalPoints.at(level).end(); iter++)
    {
        shared_ptr<Point> ptr_point = (*iter);
        if(!ptr_point->isGood)
        {
            continue;
        }
        vector<float> depthInNeighborhoods; 
        for(auto neighbor = ptr_point->indexNeighbour.begin(); neighbor != ptr_point->indexNeighbour.end(); neighbor++)
        {
            shared_ptr<Point> ptr_neighbor = refFrame->concernNormalPoints.at(level).at(*neighbor);
            if(!ptr_neighbor->isGood)
            {
                continue;
            }
            else
            {
                depthInNeighborhoods.push_back(ptr_neighbor->depth);
            }
        }
        if(depthInNeighborhoods.size() >= 3)
        {
            float result = findMedianOptimized(depthInNeighborhoods);
            ptr_point->depth = (1 - regWeight) * ptr_point->depth + regWeight * result;
        }
    }
}

void SingleTracker:: wrongOptimizeRelativePose()
{
    for(int i = LEVEL - 1; i >= 0; i--)
    {
        SE3 initialPoseForIncremental;
        if(i != LEVEL - 1)
        {
            spreadDepthToNextLevel(i);
            initialPoseForIncremental = relativaPose;
        }
        else
        {
            initialPoseForIncremental = trajectory->posesFrame.back() * relativaPose;
        }
        int iterations = 0;
        float lambda = 0.1;
        Matrix8f HThisLevel;
        Vector8f bThisLevel;
        float formerError = 0;
        float latterError = 0;
        int numLostPoints = 0;
        bool acceptance = true;
        int failedAttempt = 0;
        while(true)
        {
            if(acceptance)
            {
                // cout << "p1" << endl;
                Vector2f formerResults;
                formerResults = wrongIncrementalEquationLevel(i, relativaPose);
                formerError = formerResults[0];
                // cout << "p2" << endl;
                cout << "formerError: " << formerError << endl;
            }
            HThisLevel = HPoseAffine.at(i);
            // cout << HPoseAffine.at(i).matrix() << endl;
            for(int j = 0; j < 8; j++)
            {
                HThisLevel(j, j) *= (1 + lambda);
            }
            // cout << HSchur.at(i).matrix() << endl;
            HThisLevel -= HSchur.at(i) * (1 / (1 + lambda));
            bThisLevel = bPoseAffine.at(i) - bSchur.at(i) * (1 / (1 + lambda));
            HThisLevel = weightEquation * HThisLevel * weightEquation * (0.01 / (wG[i] * hG[i]));
            bThisLevel = weightEquation * bThisLevel * (0.01 / (wG[i] * hG[i]));

            Vector8f incrementPoseAffine;
            incrementPoseAffine = -(weightEquation * (HThisLevel.ldlt().solve(bThisLevel)));
            // cout << "look: " << endl;
            // cout << HThisLevel.matrix() << endl;
            // cout << bThisLevel.matrix() << endl;
            cout << "increment: " << endl;
            cout << incrementPoseAffine.matrix() << endl;
            // exit(0);
            relativaPoseNew = SE3::exp(incrementPoseAffine.head<6>().cast<double>()) * relativaPose;
            relativeAffineNew = relativeAffine;
            // relativeAffineNew[0] += incrementPoseAffine[6];
            // relativeAffineNew[1] += incrementPoseAffine[7];
            probeDepthThisLevel(i, lambda, incrementPoseAffine);
            
            Vector2f latterResults = checkUpdateLevel(i);
            latterError = latterResults[0];
            numLostPoints = latterResults[1];
            cout << "latterError: " << latterError << endl;
            iterations++;
            acceptance = true;
            if(latterError > formerError || numLostPoints > (int)(numPoints.at(i) * 0.2))
            {
                acceptance = false;
            }
            if(acceptance)
            {
                cout << "acceptance" << endl;
                failedAttempt = 0;
                relativaPose = relativaPoseNew;
                relativeAffine = relativeAffineNew;
                updateDepthThisLevel(i);
                smoothDepth(i);
                formerError = latterError;
            }
            else
            {
                failedAttempt += 1;
                lambda *= 10;
            }

            if(lambda > 10000 || failedAttempt >= 2 || iterations > 5)
            {
                cout << "\033[31m" << "the times of attempts is too much or lambda is too large" << "\033[0m" << endl;
                break;
            }
        }
    }
    cout << relativaPose.matrix() << endl;
    // refFrame->visualizePointCloudLevel(0);
}

Vector2f SingleTracker::wrongIncrementalEquationLevel(int level, SE3 initialPosePredicted)
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

    HSchur.at(level).setZero();
    bSchur.at(level).setZero();
    HPoseAffine.at(level).setZero();
    bSchur.at(level).setZero();

    int num = 0;
    int totalNumPoint = refFrame->concernNormalPoints.at(level).size();
    

    for(auto iter = refFrame->concernNormalPoints.at(level).begin(); iter != refFrame->concernNormalPoints.at(level).end(); iter++)
    {
        shared_ptr<Point> ptr_point = (*iter);
        float errorPoint = 0;

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
        
        bool judgement = true;

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
            if(hw < 1)
            {
                hw = sqrtf(hw);
            }
            errorPoint += hw * residual * residual * (2 - hw);
            
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
        if(!judgement)
        {
            ptr_point->status = 3;
            continue;
        }
        else
        {
            if(t.norm() < 1)
            {
                JAlphaT_mul_JAlphaSingle += 2 * 2 * totalNumPoint;
            }
            ptr_point->status = 1;
            ptr_point->capability = errorPoint;
            ptr_point->bAlpha = bAlpha;
            ptr_point->JAlphaT_mul_JAlphaSingle = JAlphaT_mul_JAlphaSingle;
            ptr_point->JAlphaT_mul_JBeta = JAlphaT_mul_JBeta;
            totalErrorPoints += errorPoint;

            HSchur.at(level) += JAlphaT_mul_JBeta * JAlphaT_mul_JBeta.transpose() / (JAlphaT_mul_JAlphaSingle + 1);
            HPoseAffine.at(level) += JBetaT_mul_JBeta;
            bSchur.at(level) += JAlphaT_mul_JBeta * bAlpha / (JAlphaT_mul_JAlphaSingle + 1);
            bPoseAffine.at(level) += bBeta;
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
    return Vector2f(totalErrorPoints, num);
}


