// ORB-SLAM3 offline sobre una carpeta de export_bag_frames.py (associations.txt, imu.csv).
//   vio_runner <vocabulario> <settings.yaml> <rgbd|rgbd_inertial|mono|mono_inertial> <carpeta> <prefijo_salida>
// Salidas:
//   <prefijo>_live.txt: por frame, lo que entrega el sistema EN ESE MOMENTO (como la odometría en vivo):
//                       "t estado tx ty tz qx qy qz qw" (pose de la cámara de color, marco óptico; estado de ORB-SLAM3:
//                       1 sin inicializar, 2 OK, 3 recién perdido, 4 perdido, 5 OK_KLT).
//   <prefijo>_final_tum.txt (RGB-D) o <prefijo>_final_euroc.txt (inerciales, pose del IMU): trayectoria final tras
//                       el ajuste global (offline).
#include <System.h>
#include <ImuTypes.h>

#include <opencv2/imgcodecs.hpp>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct Frame { double t; std::string rgb, depth; };
struct Imu { double t, wx, wy, wz, ax, ay, az; };

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "uso: vio_runner <voc> <settings> <modo> <carpeta> <prefijo>" << std::endl;
    return 1;
  }
  const std::string voc = argv[1], settings = argv[2], mode = argv[3], dir = argv[4], out = argv[5];
  const bool inertial = mode == "rgbd_inertial" || mode == "mono_inertial";
  const bool rgbd = mode == "rgbd" || mode == "rgbd_inertial";
  ORB_SLAM3::System::eSensor sensor = mode == "rgbd" ? ORB_SLAM3::System::RGBD
                                      : mode == "rgbd_inertial" ? ORB_SLAM3::System::IMU_RGBD
                                      : mode == "mono" ? ORB_SLAM3::System::MONOCULAR
                                                       : ORB_SLAM3::System::IMU_MONOCULAR;
  std::vector<Frame> frames;
  {
    std::ifstream f(dir + "/associations.txt");
    std::string line;
    while (std::getline(f, line)) {
      std::istringstream ss(line);
      Frame fr; double td;
      if (ss >> fr.t >> fr.rgb >> td >> fr.depth) frames.push_back(fr);
    }
  }
  std::vector<Imu> imu;
  if (inertial) {
    std::ifstream f(dir + "/imu.csv");
    std::string line;
    std::getline(f, line);  // cabecera
    while (std::getline(f, line)) {
      for (char& c : line) if (c == ',') c = ' ';
      std::istringstream ss(line);
      Imu m;
      if (ss >> m.t >> m.wx >> m.wy >> m.wz >> m.ax >> m.ay >> m.az) imu.push_back(m);
    }
  }
  std::cout << frames.size() << " frames, " << imu.size() << " IMU" << std::endl;
  ORB_SLAM3::System slam(voc, settings, sensor, false);
  std::ofstream live(out + "_live.txt");
  live << std::fixed;
  size_t k = 0;
  while (inertial && k < imu.size() && imu[k].t <= frames.front().t) k++;  // IMU anterior al primer frame: fuera
  for (size_t i = 0; i < frames.size(); ++i) {
    const Frame& fr = frames[i];
    cv::Mat im = cv::imread(dir + "/" + fr.rgb, cv::IMREAD_COLOR);
    if (im.empty()) { std::cerr << "no se pudo leer " << fr.rgb << std::endl; continue; }
    std::vector<ORB_SLAM3::IMU::Point> meas;
    if (inertial && i > 0)
      while (k < imu.size() && imu[k].t <= fr.t) {
        const Imu& m = imu[k++];
        meas.emplace_back(m.ax, m.ay, m.az, m.wx, m.wy, m.wz, m.t);
      }
    Sophus::SE3f tcw;
    if (rgbd) {
      cv::Mat d = cv::imread(dir + "/" + fr.depth, cv::IMREAD_UNCHANGED);
      tcw = slam.TrackRGBD(im, d, fr.t, meas);
    } else {
      tcw = slam.TrackMonocular(im, fr.t, meas);
    }
    const int state = slam.GetTrackingState();
    const Sophus::SE3f twc = tcw.inverse();
    const Eigen::Vector3f p = twc.translation();
    const Eigen::Quaternionf q = twc.unit_quaternion();
    live << std::setprecision(6) << fr.t << " " << state << " " << std::setprecision(7) << p.x() << " " << p.y()
         << " " << p.z() << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
    if (i % 500 == 0) std::cout << "frame " << i << "/" << frames.size() << " estado " << state << std::endl;
  }
  live.close();
  slam.Shutdown();
  if (inertial) slam.SaveTrajectoryEuRoC(out + "_final_euroc.txt");
  else if (rgbd) slam.SaveTrajectoryTUM(out + "_final_tum.txt");
  else slam.SaveKeyFrameTrajectoryTUM(out + "_final_kf_tum.txt");
  return 0;
}
