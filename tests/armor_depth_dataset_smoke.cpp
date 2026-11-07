#include "armor_dataset/depth_teacher.hpp"
#include <iostream>
#include <stdexcept>

int main() {
  const cv::Mat k=(cv::Mat_<double>(3,3)<<1000,0,320,0,1000,240,0,0,1);
  const cv::Mat d=cv::Mat::zeros(1,5,CV_64F);
  const std::vector<cv::Point3d> p{{-.0675,-.028,0},{.0675,-.028,0},{.0675,.028,0},{-.0675,.028,0}};
  const cv::Vec3d truth(.12,-.30,.07),translation(.08,-.05,2.8);
  std::vector<cv::Point2d> projected;cv::projectPoints(p,truth,translation,k,d,projected);
  std::array<cv::Point2f,4> c;for(int i=0;i<4;++i)c[i]=projected[i];
  const auto solved=armor_dataset::solveTeacher(c,k,d,.135,.056);
  if(!solved||cv::norm(solved->translation-translation)>1e-4)throw std::runtime_error("Teacher metric depth does not match independent synthetic projection");
  cv::Mat image=cv::Mat::zeros(480,640,CV_8UC3);
  const std::array<cv::Point2f,4> front{{{230,220},{410,220},{410,260},{230,260}}};
  cv::line(image,{230,220},{230,260},{255,80,20},8);
  cv::line(image,{410,220},{410,260},{255,80,20},8);
  cv::putText(image,"2",{292,267},cv::FONT_HERSHEY_SIMPLEX,2,{220,220,220},4);
  const auto digit=armor_dataset::findDigit(image,front);
  if(!digit||digit->box.x<260||digit->box.x+digit->box.width>380)throw std::runtime_error("Digit crop contains side lightbars or loses glyph");
  image.setTo(0);
  if(armor_dataset::findDigit(image,front))throw std::runtime_error("Empty image accepted as digit");
  cv::putText(image,"2",{300,262},cv::FONT_HERSHEY_SIMPLEX,1.7,{90,90,90},3);
  cv::rectangle(image,{285,280,70,10},{255,255,255},-1);
  if(!armor_dataset::findDigit(image,front))throw std::runtime_error("Bright background outside central plate suppresses a dim sticker digit");
  image.setTo(0);
  const std::array<cv::Point2f,4> narrow{{{300,210},{340,210},{340,270},{300,270}}};
  cv::line(image,{300,210},{300,270},{255,255,255},20);
  cv::line(image,{340,210},{340,270},{255,255,255},20);
  if(armor_dataset::findDigit(image,narrow))throw std::runtime_error("White lightbar fragments accepted as a central digit");
  std::cout<<"armor_depth_dataset_smoke passed\n";
}
