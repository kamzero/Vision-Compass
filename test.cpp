#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstring>  
#include <cstdlib>  
#include <numeric>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <dirent.h>
#include <chrono>
#include <vector>
#define PI 3.14159265358
using namespace std;
using namespace cv;

//! Camera instrinsic paras
double fx, fy, cx, cy;

//! Information for head pose of each images, especially rotation
class CameraPose {
public:
  //! Information in head_cmd from behavior
  double head_yaw;
  double head_pitch;

  //! IMU info
  double imu_yaw;

  //! Constructor
  CameraPose(double a, double b, double c):
    head_yaw(a), head_pitch(b), imu_yaw(c){}
  
  //! Print function for test
  void print(){
    cout << head_yaw << " " << head_pitch << " " << imu_yaw << endl;
  }
};

//! Calculate raw theta for keypoints
double CalRawTheta(Point2f p)
{
  return atan((p.x - cx)/fx)*180 / PI;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    cout << "usage: feature_extraction img_dir paras_file" << endl;
    return 1;
  }

  fx =  441.9120851;
  fy =  441.3849823;
  cx =  329.9198366;
  cy =  259.4488221;
  Mat ins_matrix =
    (Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

  vector<double> dist_coeff {1.3303,
                           9.93888,
                           0.000545872,
                           -0.00101315,
                           15.9781,
                           1.3529,
                           9.72562,
                           16.1226,
                           0.00283757,
                           -0.00113945,
                           -0.0021305,
                           0.00546837,
                           0.00637906,
                           -0.0305766};
  //-- Parameters
  string img_dir = argv[1];
  string paras_path = argv[2];
  
  //-- 读取图像
  DIR *dp;
  struct dirent *dirp;
  vector<string> img_names;
  vector<Mat> imgs;

	if((dp = opendir(img_dir.c_str())) == NULL)
		cout << "Can't open " << img_dir << endl;
	while((dirp = readdir(dp)) != NULL){
    //-- ignore uncommon file type
    if(dirp->d_type != 8) continue;
    img_names.push_back(img_dir + "/" + dirp->d_name);
  }
  sort(img_names.begin(), img_names.end());

  for(int i = 0; i < img_names.size(); i++){
    Mat pSrc;
    pSrc = imread(img_names[i]); 
    imgs.push_back(pSrc);
  }
	closedir(dp);

  //-- load imu_yaw && head_command
  vector<CameraPose> camera_poses;
  std::ifstream txt_in (paras_path, std::ifstream::in);
  double a, b, c;
  
  for(int i = 0; i < imgs.size(); i++){
    txt_in >> a >> b >> c;
    camera_poses.push_back( CameraPose(a,b,c) );
  }
  txt_in.close();

  //! for test: load target imgs
  CameraPose target_pose(-40, 0, 4.01789e-06);
  Mat target_img = imread("/home/kamzero/dancer-camera/hum_data/-40_0_4.01789e-06.jpg");
  vector<KeyPoint>target_keypoints;
  Mat target_descriptors;

  //-- 初始化
  vector<vector<KeyPoint>> keypoints_list(imgs.size()); 
  vector<Mat> descriptors_list(imgs.size());
  Ptr<FeatureDetector> detector = ORB::create();
  Ptr<DescriptorExtractor> descriptor = ORB::create();
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
  int max_keypoints = 0;

  //-- 第一步:检测 Oriented FAST 角点位置
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  for(int i = 0; i < imgs.size(); i++){
    detector->detect(imgs[i], keypoints_list[i]);
    max_keypoints = MAX(max_keypoints, keypoints_list[i].size());

    // Mat outimg1;
    // drawKeypoints(imgs[i], keypoints_list[i], outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    // imshow("ORB features", outimg1);
    // waitKey(0);
  }
  detector->detect(target_img, target_keypoints);

  //-- 第二步:根据角点位置计算 BRIEF 描述子
  for(int i = 0; i < imgs.size(); i++){
    descriptor->compute(imgs[i], keypoints_list[i], descriptors_list[i]);
  }
  descriptor->compute(target_img, target_keypoints, target_descriptors);

  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "extract ORB cost = " << time_used.count() << " seconds. " << endl;

  //! Calculate theta of each keypoints
  vector<vector<double>> thetas_list(imgs.size(),vector<double>(max_keypoints));
  t1 = chrono::steady_clock::now();
  for(int i = 0; i < imgs.size(); i++){
    for(int j = 0; j < keypoints_list[i].size(); j++){
      thetas_list[i][j] = CalRawTheta(keypoints_list[i][j].pt) + camera_poses[i].head_yaw + camera_poses[i].imu_yaw;
    }
    // cout << "Keypoints number for "<< i <<" th pics: "<<keypoints_list[i].size()<<endl;
  }
  t2 = chrono::steady_clock::now();
  time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "Calculate theta cost = " << time_used.count() << " seconds. " << endl;

  //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
  
  Point2f psrc, pdst;
  double theta_src, theta_dst, theta_delta;
  vector<double>delta_list;
  for(int i = 0; i < imgs.size(); i++){
    //! 如果两张图像视角相差过大，则不进行匹配
    if(abs(camera_poses[i].head_yaw + camera_poses[i].imu_yaw - 
      target_pose.head_yaw - target_pose.imu_yaw) > 50)
      continue;

    vector<DMatch> matches, good_matches;
    matcher->match(descriptors_list[i], target_descriptors, matches);

    //-- 匹配点对筛选
    // 计算最小距离和最大距离
    auto min_max = minmax_element(matches.begin(), matches.end(),
                                  [](const DMatch &m1, const DMatch &m2) { return m1.distance < m2.distance; });
    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for (int j = 0; j < descriptors_list[i].rows; j++) {
      if (matches[j].distance <= max(2 * min_dist, 30.0)) {
        good_matches.push_back(matches[j]);    
        psrc = keypoints_list[i][matches[j].queryIdx].pt;
        pdst = target_keypoints[matches[j].trainIdx].pt;
        theta_src = CalRawTheta(psrc) - camera_poses[i].head_yaw - camera_poses[i].imu_yaw;
        theta_dst = CalRawTheta(pdst) - target_pose.head_yaw - target_pose.imu_yaw;
        theta_delta = theta_src - theta_dst; 
        delta_list.push_back(theta_delta);
        // cout << CalRawTheta(psrc) << " " << theta_src << " " << CalRawTheta(pdst) << " " << theta_dst << endl;
      }
    }
    cout << camera_poses[i].head_yaw << " "<< target_pose.head_yaw << endl;
    Mat img_goodmatch;
    drawMatches(imgs[i], keypoints_list[i], target_img, target_keypoints, good_matches, img_goodmatch);
    imshow("good matches", img_goodmatch);
    waitKey(0);
  }
  double sum = accumulate(begin(delta_list), end(delta_list), 0.0);
  double mean =  sum / delta_list.size();
  cout << mean;
  return 0;
}
