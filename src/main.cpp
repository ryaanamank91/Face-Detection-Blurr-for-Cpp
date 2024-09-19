#include <opencv2/opencv.hpp>
#include "functions.hpp"

int main() {
    // Initialize the camera
    cv::VideoCapture cap = initialize_cam();

    // Load the Haar Cascade classifiers
    auto [face_cascade, profile_face_cascade] = load_haar_cascades();

    // Measure fps
    double fps_start_time = cv::getTickCount();
    int frame_count = 0;
    double fps = 0;

    while (true) {
        // Capture the frame from the camera
        cv::Mat frame;
        cap >> frame;

        if (frame.empty()) {
            std::cerr << "Error: Frame capture failed." << std::endl;
            break;
        }

        // Increment the frame count
        frame_count++;

        // Detect faces in the frame (both frontal and profile)
        auto [faces, profile_faces] = detect_faces(face_cascade, profile_face_cascade, frame);

        // Blur faces in the frame
        blur_faces(frame, faces);
        blur_faces(frame, profile_faces);

        // Calculate fps
        double fps_end_time = cv::getTickCount();
        double elapsed_time = (fps_end_time - fps_start_time) / cv::getTickFrequency();
        fps = frame_count / elapsed_time;

        // Display fps on the frame
        cv::putText(frame, "FPS: " + std::to_string(fps), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

        // Display the frame with the detected faces
        cv::imshow("Face Detection with Blurring", frame);

        // Break the loop if 'q' is pressed
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    // Release the camera and resources
    release_resources(cap);

    return 0;
}
