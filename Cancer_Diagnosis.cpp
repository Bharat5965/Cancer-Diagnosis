#include <opencv2/core/core.hpp>
#include "stdafx.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include "math.h"
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include "opencv2/objdetect/objdetect.hpp"

using namespace cv;
using namespace std;
bool finished = false;
cv::Mat img, ROI;
vector<Point> vertices;

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_RBUTTONDOWN) {
		cout << "Right mouse button clicked at (" << x << ", " << y << ")" << endl;
		if (vertices.size()<2) {
			cout << "You need a minimum of three points!" << endl;
			return;
		}

		line(img, vertices[vertices.size() - 1], vertices[0], Scalar(0, 0, 0));   // Close polygon

		Mat mask = Mat::zeros(img.rows, img.cols, CV_8UC1);               // Mask is black with white where our ROI is
		vector<vector<Point>> pts{ vertices };
		fillPoly(mask, pts, Scalar(255, 255, 255));
		img.copyTo(ROI, mask);
		finished = true;

		return;
	}
	if (event == EVENT_LBUTTONDOWN) {
		cout << "Left mouse button clicked at (" << x << ", " << y << ")" << endl;
		if (vertices.size() == 0) {
			img.at<Vec3b>(x, y) = Vec3b(255, 0, 0);                                 // First point on first click
		}
		else {
			line(img, Point(x, y), vertices[vertices.size() - 1], Scalar(0, 0, 0));    //Vertex to previous point
		}
		vertices.push_back(Point(x, y));
		return;
	}
}
int main()
{
	string fname;
	cout << "Enter the name of the image\n";
	cin >> fname;
	img = imread("C:/Users/Bharat/Documents/Visual Studio 2015/Projects/Cancer_Diagnosis/Cancer_Diagnosis/" + fname);    //Reading Image
	if (img.empty())
	{
		cout << "Error loading the image" << endl;
		exit(1);
	}
	namedWindow("Orignal Image", WINDOW_AUTOSIZE);
	imshow("Orignal Image", img);
	waitKey(50);
	int a;
	cout << "Do you want to crop the image?\nPress 1. for YES\t2. for NO.\n";
	cin >> a;
	if (a == 1) {
		setMouseCallback("Orignal Image", CallBackFunc, nullptr);   //For mouse clicks
		while (!finished) {
			imshow("Orignal Image", img);
			waitKey(50);
		}
	}
	else
	{
		destroyWindow("Orignal Image");
		ROI = img.clone();
	}

	//step 1 : map the src to the samples
	cv::Mat samples(ROI.total(), 3, CV_32F);
	auto samples_ptr = samples.ptr<float>(0);
	for (int row = 0; row != ROI.rows; ++row) {
		auto src_begin = ROI.ptr<uchar>(row);
		auto src_end = src_begin + ROI.cols * ROI.channels();
		//auto samples_ptr = samples.ptr<float>(row * src.cols);
		while (src_begin != src_end) {
			samples_ptr[0] = src_begin[0];
			samples_ptr[1] = src_begin[1];
			samples_ptr[2] = src_begin[2];
			samples_ptr += 3; src_begin += 3;
		}
	}

	//step 2 : apply kmeans to find labels and centers
	int clusterCount = 5;
	cv::Mat labels;
	int attempts = 6;
	cv::Mat centers;
	cv::kmeans(samples, clusterCount, labels,
		cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS,
			10, 0.01),
		attempts, cv::KMEANS_PP_CENTERS, centers);

	//step 3 : map the centers to the output
	cv::Mat new_image(ROI.size(), ROI.type());
	for (int row = 0; row != ROI.rows; ++row) {
		auto new_image_begin = new_image.ptr<uchar>(row);
		auto new_image_end = new_image_begin + new_image.cols * 3;
		auto labels_ptr = labels.ptr<int>(row * ROI.cols);

		while (new_image_begin != new_image_end) {
			int const cluster_idx = *labels_ptr;
			auto centers_ptr = centers.ptr<float>(cluster_idx);
			new_image_begin[0] = centers_ptr[0];
			new_image_begin[1] = centers_ptr[1];
			new_image_begin[2] = centers_ptr[2];
			new_image_begin += 3; ++labels_ptr;
		}
	}

	cv::Mat cc;
	cc = new_image.clone();
	float max[5] = { 0,0,0,0,0 };
	for (int y = 0; y < cc.rows; y++)
	{
		for (int x = 0; x < cc.cols; x++)
		{
			Vec3b intensity = cc.at<Vec3b>(y, x);
			float blue1 = intensity.val[0];
			if (blue1 <= 0) continue;
			if (max[0] == 0) {
				max[0] = blue1;
			}
			if ((max[0] != blue1) && (max[1] == 0)) {
				max[1] = blue1;
			}
			if ((max[0] != blue1) && (max[1] != blue1) && (max[2] == 0)) {
				max[2] = blue1;
			}
			if ((max[0] != blue1) && (max[1] != blue1) && (max[2] != blue1) && (max[3] == 0)) {
				max[3] = blue1;
			}
			if (max[0] == blue1) {
				max[0] = blue1;
			}
			if (max[1] == blue1) {
				max[1] = blue1;
			}if (max[2] == blue1) {
				max[2] = blue1;
			}
			if (max[3] == blue1) {
				max[3] = blue1;
			}	
		}
	}
	sort(max, max + 5);

	for (int y = 0; y < cc.rows; y++)
	{
		for (int x = 0; x < cc.cols; x++)
		{
			Vec3b intensity = cc.at<Vec3b>(y, x);
			float blue = intensity.val[0];

			if ((blue == max[0]) || (blue == max[1])) {
				intensity.val[0] = 0;
				intensity.val[1] = 0;
				intensity.val[2] = 250;
			}
			else
			{
				if ((blue == max[2]))
				{
					intensity.val[0] = 250;
					intensity.val[1] = 0;
					intensity.val[2] = 0;
				}
				else {
					intensity.val[0] = 250;
					intensity.val[1] = 250;
					intensity.val[2] = 250;
				}
			}
			cc.at<Vec3b>(Point(x, y)) = intensity;
		}
	}

	cv::namedWindow("Image", WINDOW_AUTOSIZE);
	cv::imshow("Image", ROI);
	cv::imwrite("C:/Users/Bharat/Documents/Visual Studio 2015/Projects/Cancer_Diagnosis/Cancer_Diagnosis/image.png", ROI);
	cv::namedWindow("clustered image", WINDOW_AUTOSIZE);
	cv::imshow("clustered image", new_image);
	cv::imwrite("C:/Users/Bharat/Documents/Visual Studio 2015/Projects/Cancer_Diagnosis/Cancer_Diagnosis/Image_phase_1.jpg", new_image);

	cv::imwrite("C:/Users/Bharat/Documents/Visual Studio 2015/Projects/Cancer_Diagnosis/Cancer_Diagnosis/Image_phase_2.jpg", cc);
	cv::namedWindow("Processed image", WINDOW_AUTOSIZE);
	cv::imshow("Processed image", cc);


	cv::Mat channel[3];
	cv::Mat erode_red,erode_blue,opening,closing;
	split(cc, channel);
	imwrite("C:/Users/Bharat/Documents/Visual Studio 2015/Projects/Cancer_Diagnosis/Cancer_Diagnosis/blue.png", channel[2]);
	imwrite("C:/Users/Bharat/Documents/Visual Studio 2015/Projects/Cancer_Diagnosis/Cancer_Diagnosis/red.png", channel[0]);

	int erosion_size = 1;
	float x2;
	Mat element = getStructuringElement(cv::MORPH_CROSS,
		cv::Size(1 * erosion_size + 1, 1 * erosion_size + 1),
		cv::Point(erosion_size, erosion_size));
	Mat element1 = getStructuringElement(cv::MORPH_CROSS,
		cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
		cv::Point(erosion_size, erosion_size));

	// Apply erosion or dilation on the image
	erode(channel[0], erode_red, element);  
	erode(channel[2], erode_blue, element1);
	dilate(erode_blue, opening, element);

	namedWindow("Result erode red", CV_WINDOW_AUTOSIZE);
	imshow("Result erode red",erode_red);
	namedWindow("Result erode blue", CV_WINDOW_AUTOSIZE);
	imshow("Result erode blue", erode_blue);

	namedWindow("Result erode blue opening", CV_WINDOW_AUTOSIZE);
	imshow("Result erode blue opening", opening);
	imwrite("C:/Users/Bharat/Documents/Visual Studio 2015/Projects/Cancer_Diagnosis/Cancer_Diagnosis/result_erode_blue_opening.png", opening);
	imwrite("C:/Users/Bharat/Documents/Visual Studio 2015/Projects/Cancer_Diagnosis/Cancer_Diagnosis/erode_red.png", erode_red);
	imwrite("C:/Users/Bharat/Documents/Visual Studio 2015/Projects/Cancer_Diagnosis/Cancer_Diagnosis/erode_blue.png", erode_blue);

	//blob detection for red
	SimpleBlobDetector::Params params;
	params.minThreshold = 5;
	params.maxThreshold = 256;
	// Filter by Area.
	params.filterByArea = true;
	params.minArea = 10;
	params.maxArea = 700;
	//filter by colour
	params.filterByColor = true;
	params.blobColor = 0;
	// Storage for blobs
	vector<KeyPoint> keypoints;
	// Set up detector with params
	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
	// Detect blobs
	detector->detect(opening, keypoints);
	//the total no of blobs detected are:
	size_t x = keypoints.size();
	cout << "total no of blue circles detected are:" << x << endl;
	//blob detection for blue
	SimpleBlobDetector::Params params1;
	params1.minThreshold = 1;
	params1.maxThreshold = 256;
	// Filter by Area.
	params1.filterByArea = true;
	params1.minArea =5;
	params1.maxArea = 700;
	//filter by colour
	params1.filterByColor = true;
	params1.blobColor = 0;
	// Storage for blobs
	vector<KeyPoint> keypoints1;
	// Set up detector with params
	Ptr<SimpleBlobDetector> detector1 = SimpleBlobDetector::create(params1);
	// Detect blobs
	detector1->detect(erode_red, keypoints1);
	//the total no of blobs detected are:
	size_t x1 = keypoints1.size();
	cout << "total no of red circles detected are:" << x1 << endl;
	
	x2 = (x1*100) / (x1 + x);
	cout << "Proliferative Index: " << x2<<"%"<<endl;
	cv::waitKey();
	return 0;
}