#include "functions.hpp"

cv::VideoCapture initialize_cam() {
    cv::VideoCapture cap(2);
    if (!cap.isOpened()) {
        throw std::runtime_error("Error: Could not open the camera.");
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    return cap;
}

std::pair<cv::CascadeClassifier, cv::CascadeClassifier> load_haar_cascades() {
    cv::CascadeClassifier face_cascade;
    cv::CascadeClassifier profile_face_cascade;

    std::string face_cascade_path = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml";
    std::string profile_face_cascade_path = "/usr/share/opencv4/haarcascades/haarcascade_profileface.xml";

    if (!face_cascade.load(face_cascade_path) || !profile_face_cascade.load(profile_face_cascade_path)) {
        throw std::runtime_error("Error: Could not load Haar Cascade classifiers.");
    }

    return {face_cascade, profile_face_cascade};
}

std::pair<std::vector<cv::Rect>, std::vector<cv::Rect>> detect_faces(cv::CascadeClassifier& face_cascade, cv::CascadeClassifier& profile_face_cascade, const cv::Mat& frame) {
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    std::vector<cv::Rect> faces;
    std::vector<cv::Rect> profile_faces;

    face_cascade.detectMultiScale(gray, faces, 1.05, 5, 0, cv::Size(50, 50), cv::Size(300, 300));
    profile_face_cascade.detectMultiScale(gray, profile_faces, 1.05, 5, 0, cv::Size(50, 50), cv::Size(300, 300));

    cv::Mat flipped_gray;
    cv::flip(gray, flipped_gray, 1);

    std::vector<cv::Rect> left_profile_faces;
    profile_face_cascade.detectMultiScale(flipped_gray, left_profile_faces, 1.05, 5, 0, cv::Size(50, 50), cv::Size(300, 300));

    for (auto& rect : left_profile_faces) {
        rect.x = gray.cols - rect.x - rect.width;
        profile_faces.push_back(rect);
    }

    return {faces, profile_faces};
}

void blur_faces(cv::Mat& frame, const std::vector<cv::Rect>& faces) {
    for (const auto& face : faces) {
        cv::Mat face_roi = frame(face);
        cv::GaussianBlur(face_roi, face_roi, cv::Size(21, 21), 30);
    }
}

void release_resources(cv::VideoCapture& cap) {
    cap.release();
    cv::destroyAllWindows();
}
