#include "armor_dataset/depth_teacher.hpp"
#include "runtime/auto_aim_config.hpp"
#include "runtime/armor_detector_factory.hpp"
#include "sensor/camera/camera_calibration.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>

#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {
const char* keys =
  "{help h | false | Show help}"
  "{config | config/auto_aim.yaml | Detector and small-armor geometry}"
  "{calibration c | config/camera_config.yaml | Calibration belonging to this recording}"
  "{model | model/armor_model/yolov5.xml | Teacher armor model}"
  "{device | CPU | Inference device}"
  "{output o | out/armor_depth_teacher | Empty output directory}"
  "{split | train | train/val/test; assign entire recordings separately}"
  "{sequence | | Recording ID, defaults to video stem}"
  "{stride | 9 | Sequential decode, one sampled frame per N frames}"
  "{max-frames | 0 | Stop after N decoded frames; 0 means full video}"
  "{max-rmse | 1.5 | Maximum teacher reprojection RMSE in pixels}"
  "{min-light | 10 | Minimum refined lightbar length in pixels}"
  "{min-depth | 0.3 | Minimum optical-axis depth in metres}"
  "{max-depth | 15 | Maximum optical-axis depth in metres}"
  "{@video | video/3m_high.avi | Video file}";

void require(bool ok,const std::string& reason) {if(!ok)throw std::runtime_error(reason);}
json matrixJson(const cv::Mat& matrix) {
  json result=json::array();
  for(int r=0;r<matrix.rows;++r){json row=json::array();for(int c=0;c<matrix.cols;++c)row.push_back(matrix.at<double>(r,c));result.push_back(row);}
  return result;
}
}

int main(int argc,char** argv) try {
  cv::CommandLineParser parser(argc,argv,keys);
  if(parser.get<bool>("help")){parser.printMessage();return 0;}
  const fs::path video_path=fs::absolute(parser.get<std::string>(0));
  const fs::path output=fs::absolute(parser.get<std::string>("output"));
  const std::string split=parser.get<std::string>("split");
  const std::string requested_sequence=parser.get<std::string>("sequence");
  const std::string sequence=requested_sequence.empty()?video_path.stem().string():requested_sequence;
  const int stride=parser.get<int>("stride"),max_frames=parser.get<int>("max-frames");
  const double max_rmse=parser.get<double>("max-rmse"),min_light=parser.get<double>("min-light");
  const double min_depth=parser.get<double>("min-depth"),max_depth=parser.get<double>("max-depth");
  require(parser.check(),"Invalid CLI arguments");
  require(stride>0 && max_frames>=0 && min_depth>0 && max_depth>min_depth,"Invalid sampling/depth range");
  require(split=="train"||split=="val"||split=="test","split must be train/val/test");
  require(sequence.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")==std::string::npos,"Unsafe sequence ID");
  require(!fs::exists(output)||fs::is_empty(output),"Output must be empty; existing samples will not be overwritten");
  auto config=runtime::loadAutoAimConfig(parser.get<std::string>("config"));
  const auto calibration=L1Sensor::loadCameraCalibration(
    YAML::LoadFile(parser.get<std::string>("calibration"))["calibration"],parser.get<std::string>("calibration"));
  const auto refine_config=config.refiner;
  config.refiner.enable=false;
  config.inference.model_path=parser.get<std::string>("model");
  config.inference.device=parser.get<std::string>("device");
  auto detector=runtime::makeDetector(config,true);
  auto enabled_refiner=refine_config; enabled_refiner.enable=true;
  L2Perception::ArmorRefiner refiner(enabled_refiner);
  cv::VideoCapture capture(video_path.string());
  require(capture.isOpened(),"Cannot open video");
  const double fps=capture.get(cv::CAP_PROP_FPS);
  require(std::isfinite(fps)&&fps>0,"Video has no valid FPS");
  fs::create_directories(output/"crops"/split);
  fs::create_directories(output/"review");
  fs::create_directories(output/"review_rejected");
  std::ofstream manifest(output/"manifest.jsonl");
  std::ofstream rejected_manifest(output/"rejected_manifest.jsonl");
  std::map<std::string,int> rejected;
  int decoded=0,sampled=0,saved=0,rejected_reviews=0;
  std::vector<double> depths;
  cv::Mat image;
  while((max_frames==0||decoded<max_frames) && capture.read(image)) {
    const int frame_index=decoded++;
    require(calibration.matchesImageSize(image.size()),"Recording size differs from calibration; refusing fabricated intrinsics");
    if(frame_index%stride!=0)continue;
    ++sampled;
    auto detections=detector.detect(image);
    std::vector<L2Perception::RefineRecord> records;
    refiner.refine(image,detections,&records);
    for(std::size_t index=0;index<detections.size();++index) {
      const auto& armor=detections[index];
      // Old recordings can contain physically large hero armor. Do not label it
      // using the small dimensions requested for this dataset.
      if(armor.class_id<2||armor.class_id>5 || L2Perception::isLargeArmor(L2Perception::armorClassFromId(armor.class_id)).value_or(true)) {++rejected["not_small_numeric_class"];continue;}
      if(index>=records.size()||records[index].verdict!=L2Perception::RefineVerdict::Refined){++rejected["no_verified_light_pair"];continue;}
      const double length=.5*(cv::norm(armor.corners[3]-armor.corners[0])+cv::norm(armor.corners[2]-armor.corners[1]));
      if(length<min_light){++rejected["light_too_small"];continue;}
      const auto pose=armor_dataset::solveTeacher(armor.corners,calibration.camera_matrix,
        calibration.distortion_coefficients,config.armor.small_width,config.armor.height);
      if(!pose||pose->rmse_px>max_rmse){++rejected["pnp_or_reprojection"];continue;}
      if(pose->translation[2]<min_depth||pose->translation[2]>max_depth){++rejected["depth_range"];continue;}
      if(pose->alternative_depth_relative_gap>.10){++rejected["ambiguous_depth"];continue;}
      const auto digit=armor_dataset::findDigit(image,armor.corners);
      if(!digit){
        ++rejected["digit_crop_unreliable"];
        if(rejected_reviews<8){
          const auto guide=armor_dataset::digitGuide(armor.corners);
          json rejected_corners=json::array();
          for(const auto& p:armor.corners)rejected_corners.push_back({p.x,p.y});
          rejected_manifest<<json{{"frame_index",frame_index},{"corners",rejected_corners}}.dump()<<'\n';
          const cv::Rect context=(cv::boundingRect(std::vector<cv::Point2f>(guide.begin(),guide.end()))+
            cv::Size(80,80)-cv::Point(40,40))&cv::Rect(0,0,image.cols,image.rows);
          cv::Mat view=image(context).clone();std::vector<cv::Point> outline;
          for(const auto& p:guide)outline.emplace_back(cvRound(p.x-context.x),cvRound(p.y-context.y));
          cv::polylines(view,outline,true,{0,255,255},1);
          cv::imwrite((output/"review_rejected"/("f"+std::to_string(frame_index)+"_a"+std::to_string(index)+".jpg")).string(),view);
          ++rejected_reviews;
        }
        continue;
      }
      const auto& box=digit->box;
      const std::string stem=sequence+"_f"+std::to_string(frame_index)+"_a"+std::to_string(index);
      const std::string crop="crops/"+split+"/"+stem+".png";
      require(cv::imwrite((output/crop).string(),image(box)),"Cannot save digit crop");
      json corners=json::array(); for(const auto& p:armor.corners)corners.push_back({p.x,p.y});
      json distortion=json::array();for(std::size_t i=0;i<calibration.distortion_coefficients.total();++i)distortion.push_back(calibration.distortion_coefficients.at<double>(static_cast<int>(i)));
      const double depth=pose->translation[2],range=cv::norm(pose->translation);
      manifest<<json{{"id",stem},{"split",split},{"sequence",sequence},{"source_video",video_path.string()},
        {"frame_index",frame_index},{"time_s_nominal",frame_index/fps},{"digit",armor.class_id},
        {"digit_label_source","teacher_network_class_id"},{"crop",crop},{"digit_bbox_xyxy",{box.x,box.y,box.x+box.width,box.y+box.height}},
        {"image_size",{image.cols,image.rows}},{"camera_matrix",matrixJson(calibration.camera_matrix)},
        {"distortion_coefficients",distortion},{"teacher_corners_xy",corners},
        {"teacher_rvec",{pose->rotation[0],pose->rotation[1],pose->rotation[2]}},
        {"xyz_camera_m",{pose->translation[0],pose->translation[1],pose->translation[2]}},
        {"depth_z_m",depth},{"range_m",range},{"panel_type","small"},
        {"panel_size_m",{config.armor.small_width,config.armor.height}},
        {"label_source","lightbar_refined_ippe_pseudo_label"},{"teacher_rmse_px",pose->rmse_px},
        {"teacher_alternative_depth_relative_gap",pose->alternative_depth_relative_gap},
        {"light_length_px",length},{"digit_contrast",digit->contrast},{"detection_confidence",armor.confidence}}.dump()<<'\n';
      depths.push_back(depth);
      if(saved<36){
        const cv::Rect context=(box+cv::Size(80,80)-cv::Point(40,40))&cv::Rect(0,0,image.cols,image.rows);
        cv::Mat review=image(context).clone();
        cv::rectangle(review,cv::Rect(box.x-context.x,box.y-context.y,box.width,box.height),{0,255,0},1);
        cv::imwrite((output/"review"/(stem+".jpg")).string(),review);
      }
      ++saved;
    }
    if(sampled%50==0)std::cout<<sequence<<" decoded="<<decoded<<" sampled="<<sampled<<" saved="<<saved<<std::endl;
  }
  std::sort(depths.begin(),depths.end());
  const json summary{{"sequence",sequence},{"split",split},{"source_video",video_path.string()},
    {"decoded_frames",decoded},{"sampled_frames",sampled},{"samples",saved},{"rejections",rejected},
    {"decode_end","VideoCapture.read returned false or max-frames limit; this is not a file-integrity guarantee"},
    {"teacher_calibration",fs::absolute(parser.get<std::string>("calibration")).string()},
    {"teacher_model",fs::absolute(parser.get<std::string>("model")).string()},
    {"small_geometry_m",{config.armor.small_width,config.armor.height}},
    {"depth_min_m",saved?json(depths.front()):json(nullptr)},
    {"depth_max_m",saved?json(depths.back()):json(nullptr)},
    {"label_status","pseudo_labels_not_independent_range_ground_truth"},
    {"input_contract","raw digit crop + digit bbox + camera intrinsics; teacher corners are excluded"}};
  std::ofstream(output/"summary.json")<<summary.dump(2)<<'\n';
  std::cout<<summary.dump(2)<<std::endl;
  require(saved>0,"No reliable samples exported; inspect teacher/calibration/refiner/crop quality");
  return 0;
} catch(const std::exception& error) {std::cerr<<"armor_depth_export: "<<error.what()<<'\n';return 1;}
