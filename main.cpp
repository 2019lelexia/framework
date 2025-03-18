#include "process_image.h"
#include "frame.h"
#include "pixel_select.h"
#include "tracker.h"
#include "trajectory.h"
#include "map_point.h"
#include "map.h"



int main()
{
    // shared_ptr<ImageInfo> image = make_shared<ImageInfo>();
    // image->readImage("./ok.png");
    // image->makePyramids();
    // image->setGlobalSize();
    // image->calculateGrad();
    // cout << image->dx[2].at<float>(10, 20) << endl;

    shared_ptr<ImageFolder> folder = make_shared<ImageFolder>("./", "../calibration/kitti.xml", "./times.txt");
    folder->readImageFolder();
    folder->readTimestamps();

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

    shared_ptr<Frame> ref_frame = make_shared<Frame>();
    ref_frame->setFrame(folder->getIndice(0));

    shared_ptr<Frame> tar_frame = make_shared<Frame>();
    tar_frame->setFrame(folder->getIndice(1));

    shared_ptr<Frame> tar_frame2 = make_shared<Frame>();
    tar_frame2->setFrame(folder->getIndice(2));

    shared_ptr<Frame> tar_frame3 = make_shared<Frame>();
    tar_frame3->setFrame(folder->getIndice(3));

    shared_ptr<Frame> tar_frame4 = make_shared<Frame>();
    tar_frame4->setFrame(folder->getIndice(4));

    shared_ptr<Frame> tar_frame5 = make_shared<Frame>();
    tar_frame5->setFrame(folder->getIndice(5));

    shared_ptr<Frame> tar_frame6 = make_shared<Frame>();
    tar_frame6->setFrame(folder->getIndice(6));

    shared_ptr<Frame> tar_frame7 = make_shared<Frame>();
    tar_frame7->setFrame(folder->getIndice(7));

    shared_ptr<PixelSelector> selector = make_shared<PixelSelector>();
    selector->setParameters(40, 20, 10, 3, 3, 20, 10, 2000, 40000, 1, 1);
    selector->selectKeyPointFromImage(ref_frame);

    // for(auto iter = frame->keyPoints[3].begin(); iter != frame->keyPoints[3].end(); iter++)
    // {
    //     cout << iter->pt.x << " " << iter->pt.y << endl;
    // }
    // selector->lookKeyPoint(ref_frame);

    // selector->selectNormalPointFromImage(ref_frame);
    selector->selectNormalPointEvenly(ref_frame);
    // for(auto iter = frame->normalPoints[0].begin(); iter != frame->normalPoints[0].end(); iter++)
    // {
    //     cout << iter->x << " " << iter->y << endl;
    // }
    selector->lookNormalPoint(ref_frame);

    ref_frame->transformToConcern();

    shared_ptr<Trajectory> trajectoryer = make_shared<Trajectory>();
    shared_ptr<Map> globalMap = make_shared<Map>();
    shared_ptr<SingleTracker> tracker = make_shared<SingleTracker>();
    tracker->setTrajectory(trajectoryer);
    // cout << tracker->relativaPose.matrix() << endl;

    tracker->setRefFrame(ref_frame);
    tracker->setTarFrame(tar_frame);
    
    ref_frame->makePointCloudFlann();
    cout << "check point cloud flann ok" << endl;

    // tracker->wrongOptimizeRelativePose();
    // tracker->optimizeRelativePose();

    // cout << "the first one: " << endl;
    // cout << tracker->relativaPose.matrix() << endl;
    trajectoryer->addPoseAndAffineOfTrajectory(1, tracker->relativaPose, AffineLight(tracker->relativeAffine[0], tracker->relativeAffine[1]));
    // ref_frame->visualizePointCloudLevel(0);

    tracker->setTarFrame(tar_frame2);
    tracker->optimizeDSO();
    // tracker->optimizeRelativePose();
    // tracker->optimizeRelativePose2ml();
    cout << "the second one: " << endl;
    // cout << tracker->relativaPose.matrix() << endl;
    trajectoryer->addPoseAndAffineOfTrajectory(2, tracker->relativaPose, AffineLight(tracker->relativeAffine[0], tracker->relativeAffine[1]));
    // ref_frame->visualizePointCloudLevel(0);

    tracker->setTarFrame(tar_frame3);
    tracker->optimizeDSO();
    // tracker->optimizeRelativePose();
    cout << "the third one: " << endl;
    // cout << tracker->relativaPose.matrix() << endl;
    trajectoryer->addPoseAndAffineOfTrajectory(3, tracker->relativaPose, AffineLight(tracker->relativeAffine[0], tracker->relativeAffine[1]));
    // ref_frame->visualizePointCloudLevel(0);

    tracker->setTarFrame(tar_frame4);
    tracker->optimizeDSO();
    // tracker->optimizeRelativePose();
    cout << "the forth one: " << endl;
    // cout << tracker->relativaPose.matrix() << endl;
    trajectoryer->addPoseAndAffineOfTrajectory(4, tracker->relativaPose, AffineLight(tracker->relativeAffine[0], tracker->relativeAffine[1]));
    // ref_frame->visualizePointCloudLevel(0);

    tracker->setTarFrame(tar_frame5);
    tracker->optimizeDSO();
    // tracker->optimizeRelativePose();
    trajectoryer->addPoseAndAffineOfTrajectory(5, tracker->relativaPose, AffineLight(tracker->relativeAffine[0], tracker->relativeAffine[1]));
    cout << "the fifth one: " << endl;
    // cout << tracker->relativaPose.matrix() << endl;

    tracker->setTarFrame(tar_frame6);
    tracker->optimizeDSO();
    // tracker->optimizeRelativePose();
    trajectoryer->addPoseAndAffineOfTrajectory(6, tracker->relativaPose, AffineLight(tracker->relativeAffine[0], tracker->relativeAffine[1]));
    cout << "the sixth one: " << endl;
    // cout << tracker->relativaPose.matrix() << endl;

    tracker->setTarFrame(tar_frame7);
    tracker->optimizeDSO();
    // tracker->optimizeRelativePose();
    trajectoryer->addPoseAndAffineOfTrajectory(7, tracker->relativaPose, AffineLight(tracker->relativeAffine[0], tracker->relativeAffine[1]));
    cout << "the seventh one: " << endl;
    cout << tracker->positionMutation << endl;
    cout << tracker->relativaPose.matrix() << endl;

    if(tracker->positionMutation >= 5)
    {
        tracker->finishInitialization = true;
    }
    if(tracker->finishInitialization)
    {
        trajectoryer->trackingFrames.push_back(ref_frame);
        float totalDepth = 1e-5;
        float totalNumber = 1e-5;
        for(auto iter = ref_frame->concernNormalPoints.at(0).begin(); iter != ref_frame->concernNormalPoints.at(0).end(); iter++)
        {
            shared_ptr<Point> ptr_point = (*iter);
            totalDepth += ptr_point->depthConvergence;
            totalNumber += 1;
        }
        float scaleFactor = 1 / (totalDepth / totalNumber);
        float selectPercentage = initializationTransformNum / ref_frame->concernNormalPoints.at(0).size();
        cout << "initialization over, now transform initial points into map point." << endl << "We keep " << selectPercentage * 100 << "\% points here." << endl;
        for(auto iter = ref_frame->concernNormalPoints.at(0).begin(); iter != ref_frame->concernNormalPoints.at(0).end(); iter++)
        {
            if(rand() / (float) RAND_MAX > selectPercentage)
            {
                continue;
            }
            shared_ptr<Point> ptr_point = (*iter);
            shared_ptr<MapPoint> ptr_mappoint = make_shared<MapPoint>(ref_frame, (float)ptr_point->positionX, (float)ptr_point->positionY);
            ptr_mappoint->immaturePoint = make_shared<Immature>(ref_frame, ptr_mappoint);
            if(!isfinite(ptr_mappoint->immaturePoint->energyTH))
            {
                ptr_mappoint->immaturePoint->map_point = nullptr;
                ptr_mappoint->immaturePoint = nullptr;
                continue;
            }
            ptr_mappoint->createFromImmaturePoint();
            if(!isfinite(ptr_mappoint->energyTH))
            {
                ptr_mappoint->deleteThisPoint();
                continue;
            }
            ref_frame->mapPoints.push_back(ptr_mappoint);
            ptr_mappoint->setDepthScaled(ptr_point->depthConvergence * scaleFactor);
            ptr_mappoint->setDepthZero(ptr_mappoint->depth);
            ptr_mappoint->ownDepthPrior = true;
            ptr_mappoint->stateOptimize = MapPoint::PointStatus::ACTIVE;
            ptr_mappoint->delta = ptr_mappoint->depth - ptr_mappoint->depthZero;
        }
        SE3 initialToNew = tracker->relativaPose;
        initialToNew.translation() /= scaleFactor;
        ref_frame->setPoseAndStateScaledInitially(ref_frame->Tw_to_c, ref_frame->affine);
        tar_frame7->Tw_to_c = initialToNew;
        tar_frame7->setPoseAndStateScaledInitially(tar_frame7->Tw_to_c, tar_frame7->affine);
        
        globalMap->addKeyFrameToMap(ref_frame);
        cout << "initialization is all over" << endl;
    }

    // ref_frame->visualizePointCloudLevel(3, tracker->relativaPose);

    // trajectoryer->visualizeTrajectory();

    return 0;
    

}