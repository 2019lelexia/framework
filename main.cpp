#include "process_image.h"
#include "frame.h"
#include "pixel_select.h"
#include "tracker.h"
#include "trajectory.h"



int main()
{
    // shared_ptr<ImageInfo> image = make_shared<ImageInfo>();
    // image->readImage("./ok.png");
    // image->makePyramids();
    // image->setGlobalSize();
    // image->calculateGrad();
    // cout << image->dx[2].at<float>(10, 20) << endl;

    shared_ptr<ImageFolder> folder = make_shared<ImageFolder>("./", "../calibration/kitti.xml");
    folder->readImageFolder();

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
    cout << tracker->relativaPose.matrix() << endl;
    trajectoryer->addPoseAndAffineOfTrajectory(2, tracker->relativaPose, AffineLight(tracker->relativeAffine[0], tracker->relativeAffine[1]));
    // ref_frame->visualizePointCloudLevel(0);

    tracker->setTarFrame(tar_frame3);
    tracker->optimizeDSO();
    // tracker->optimizeRelativePose();
    cout << "the third one: " << endl;
    cout << tracker->relativaPose.matrix() << endl;
    trajectoryer->addPoseAndAffineOfTrajectory(3, tracker->relativaPose, AffineLight(tracker->relativeAffine[0], tracker->relativeAffine[1]));
    // ref_frame->visualizePointCloudLevel(0);

    tracker->setTarFrame(tar_frame4);
    tracker->optimizeDSO();
    // tracker->optimizeRelativePose();
    cout << "the forth one: " << endl;
    cout << tracker->relativaPose.matrix() << endl;
    trajectoryer->addPoseAndAffineOfTrajectory(4, tracker->relativaPose, AffineLight(tracker->relativeAffine[0], tracker->relativeAffine[1]));
    // ref_frame->visualizePointCloudLevel(0);

    tracker->setTarFrame(tar_frame5);
    tracker->optimizeDSO();
    // tracker->optimizeRelativePose();
    trajectoryer->addPoseAndAffineOfTrajectory(5, tracker->relativaPose, AffineLight(tracker->relativeAffine[0], tracker->relativeAffine[1]));
    cout << "the fifth one: " << endl;
    cout << tracker->relativaPose.matrix() << endl;

    tracker->setTarFrame(tar_frame6);
    tracker->optimizeDSO();
    // tracker->optimizeRelativePose();
    trajectoryer->addPoseAndAffineOfTrajectory(6, tracker->relativaPose, AffineLight(tracker->relativeAffine[0], tracker->relativeAffine[1]));
    cout << "the sixth one: " << endl;
    cout << tracker->relativaPose.matrix() << endl;

    ref_frame->visualizePointCloudLevel(3);

    trajectoryer->visualizeTrajectory();

    return 0;
    

}