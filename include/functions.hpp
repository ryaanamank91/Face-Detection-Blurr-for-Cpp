#ifndef FUNCTIONS_HPP
#define FUNCTIONS_HPP

#include <opencv2/opencv.hpp>

cv::VideoCapture initialize_cam();
std::pair<cv::CascadeClassifier, cv::CascadeClassifier> load_haar_cascades();
std::pair<std::vector<cv::Rect>, std::vector<cv::Rect>> detect_faces(cv::CascadeClassifier& face_cascade, cv::CascadeClassifier& profile_face_cascade, const cv::Mat& frame);
void blur_faces(cv::Mat& frame, const std::vector<cv::Rect>& faces);
void release_resources(cv::VideoCapture& cap);

#endif // FUNCTIONS_HPP
