
#include "Utilities.h"
#include <opencv2/opencv.hpp>

void loadImage(Mat* image, const std::string& path)
{
	*image = imread(path, -1);

	if (image->empty())
	{
		std::cout << "Could not open " << path << std::endl;
	}
}

void perspectiveTransform(Mat* src, Mat* dst, cv::Point2f* srcPoints, size_t width, size_t height)
{
	// NW, NE, SE, SW
	cv::Point2f dstPoints[4] = {
		{0.0f,          0.0f},
		{width - 1.0f,   0.0f},
		{width - 1.0f,   height - 1.0f},
		{0.0f,           height - 1.0f}
	};

	cv::Mat H = cv::getPerspectiveTransform(srcPoints, dstPoints);

	cv::warpPerspective(*src, *dst, H, cv::Size(width, height));
}

std::vector<cv::Vec3f> detectBallsX(Mat* image)
{
	cv::Mat greenMask;
	{
		cv::Mat hsv;
		cv::cvtColor(*image, hsv, cv::COLOR_BGR2HSV);

		cv::inRange(
			hsv,
			cv::Scalar(40, 40, 65),
			cv::Scalar(90, 255, 255),
			greenMask
		);

		//cv::Mat candidateMask;
		//cv::bitwise_not(greenMask, candidateMask);
		//
		//cv::morphologyEx(
		//	candidateMask,
		//	candidateMask,
		//	cv::MORPH_CLOSE,
		//	cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5))
		//);
	}

	cv::Mat gray;
	{
		cv::cvtColor(*image, gray, cv::COLOR_BGR2GRAY);

		const int kernelSize = 5;
		const double strength = 1.2;
		cv::GaussianBlur(gray, gray, cv::Size(kernelSize, kernelSize), strength);
		gray.setTo(0, greenMask);
	}

	imshow(".", gray);

	std::vector<cv::Vec3f> circles;
	{
		const double inverseAccumReso = 1.2;
		const double minCenterDistance = 33.0;
		const double upperCannyThreshold = 50;
		const double accumThreshold = 15.0;
		const int minRadius = 25;
		const int maxRadius = 32;

		cv::HoughCircles(
			gray,
			circles,
			cv::HOUGH_GRADIENT,
			inverseAccumReso,
			minCenterDistance,
			upperCannyThreshold,
			accumThreshold,
			minRadius,
			maxRadius
		);
	}

	// TODO: Reject if on felt or net

	return circles;
}

std::vector<cv::Vec3f> detectBalls(cv::Mat* image)
{
	CV_Assert(image != nullptr && !image->empty());

	cv::Mat foreground;
	{
		cv::Mat hsv;
		cv::cvtColor(*image, hsv, cv::COLOR_BGR2HSV);

		cv::Mat greenMask;
		cv::inRange(
			hsv,
			cv::Scalar(40, 60, 60),
			cv::Scalar(90, 255, 255),
			greenMask
		);

		cv::bitwise_not(greenMask, foreground);

		cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
		cv::morphologyEx(foreground, foreground, cv::MORPH_CLOSE, kernel);
		cv::morphologyEx(foreground, foreground, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
	}

	cv::Mat dist;
	{
		cv::distanceTransform(foreground, dist, cv::DIST_L2, 5);

		cv::Mat dist8;
		cv::normalize(dist, dist8, 0, 255, cv::NORM_MINMAX);
		dist8.convertTo(dist8, CV_8U);

		//imshow(".", dist8);
	}

	// Tune this fraction: higher = fewer, more central seeds.
	cv::Mat sureForeground;
	{
		cv::threshold(
			dist,
			sureForeground,
			0.7 * 30.0,  // Assumes ball radius around 30 px
			255,
			cv::THRESH_BINARY
		);
		sureForeground.convertTo(sureForeground, CV_8U);
	}

	//imshow(".", sureForeground);

	cv::Mat seedLabels;
	int nSeeds = cv::connectedComponents(sureForeground, seedLabels, 8, CV_32S);

	cv::Mat markers = cv::Mat::zeros(foreground.size(), CV_32S);
	markers.setTo(1, foreground == 0);

	for (int y = 0; y < seedLabels.rows; ++y)
	{
		for (int x = 0; x < seedLabels.cols; ++x)
		{
			int label = seedLabels.at<int>(y, x);

			if (label > 0)
				markers.at<int>(y, x) = label + 1;
		}
	}

	cv::Mat watershedImage = image->clone();
	cv::watershed(watershedImage, markers);

	cv::Mat display;
	cv::normalize(
		seedLabels,
		display,
		0, 255,
		cv::NORM_MINMAX,
		CV_8U
	);

	//imshow(".", display);

	std::vector<cv::Vec3f> circles;

	for (int label = 2; label < nSeeds + 1; ++label)
	{
		cv::Mat region = (markers == label);

		std::vector<cv::Point> points;
		cv::findNonZero(region, points);

		if (points.size() < 20)
			continue;

		cv::Point2f center;
		float radius;
		cv::minEnclosingCircle(points, center, radius);

		// Reject implausible ball sizes.
		if (radius >= 20.0f && radius <= 40.0f)
		{
			circles.emplace_back(
				center.x, center.y, radius);
		}
	}

	return circles;
}

// For Debugging
void drawHollowCircles(
	Mat* image,
	const std::vector<cv::Vec3f>& circles,
	const cv::Scalar& color = cv::Scalar(0, 0, 255),
	int thickness = 2)
{
	for (const auto& circle : circles)
	{
		const cv::Point2i p = cv::Point2i(circle[0], circle[1]);
		const float radius = circle[2];
		cv::circle(*image, p, radius, color, thickness);
	}
}

void nearestNeighbourQuantize(Mat* src, Mat* dst)
{
	// BGR
	std::vector<cv::Vec3f> paletteF = {
		{0.5, 0.8, 0.9},		// White
		{0, 0.5, 0.9},			// Yellow
		{0.3, 0, 0},			// Blue
		{0, 0, 1.0},			// Red
		{0.1, 0, 0.05},			// Purple
		{0, 0.15, 0.9},			// Orange
		{0.1, 0.2, 0.1},		// Green
		{0.05, 0.05, 0.3},		// Brown
		{0, 0, 0},				// Black
		{0.36, 0.5, 0.34},		// Green Table
	};

	std::vector<cv::Vec3b> paletteU;
	for (size_t i = 0; i < paletteF.size(); i++)
	{
		cv::Vec3b colorU = cv::Vec3b(255 * paletteF[i][0], 255 * paletteF[i][1], 255 * paletteF[i][2]);
		paletteU.emplace_back(colorU);
	}

	size_t width = src->cols;
	size_t height = src->rows;

	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			cv::Vec3b pixel = src->at<cv::Vec3b>(y, x);

			size_t bestIndex = 0;
			int minLenSqr = INT_MAX;

			for (size_t i = 0; i < paletteU.size(); ++i)
			{
				int db = static_cast<int>(pixel[0]) - paletteU[i][0];
				int dg = static_cast<int>(pixel[1]) - paletteU[i][1];
				int dr = static_cast<int>(pixel[2]) - paletteU[i][2];

				size_t lenSqr = db * db + dg * dg + dr * dr;

				if (lenSqr < minLenSqr)
				{
					minLenSqr = lenSqr;
					bestIndex = i;
				}
			}

			dst->at<cv::Vec3b>(y, x) = paletteU[bestIndex];
		}
	}
}

void MyApplication()
{
	const char* folderPath = "Media/Pool/";
	const char* fileNames[] = {
		"AllBalls.jpeg",
		"Table1.jpeg",
		"Table2.jpeg",
		"Table3.jpeg",
		"Table4.jpeg",
		"Table5.jpeg",
		"Table6.jpeg"
	};

	int imageCount = sizeof(fileNames) / sizeof(fileNames[0]);
	Mat* inputImages = new Mat[imageCount];
	Mat* outputImages = new Mat[imageCount];

	for (int i = 0; i < imageCount; i++)
	{
		Mat* in = &inputImages[i];
		Mat* out = &outputImages[i];

		std::string filePath = std::string(folderPath) + fileNames[i];
		loadImage(in, filePath);

		// NW, NE, SE, SW
		cv::Point2f srcPoints[4] = {
			{337,  35},
			{1570, 21},
			{1888, 451},
			{38, 472},
		};

		size_t width = in->cols;
		size_t height = in->rows;

		perspectiveTransform(in, out, srcPoints, width, height);

		std::vector<cv::Vec3f> circles = detectBalls(out);
		drawHollowCircles(out, circles);

		//nearestNeighbourQuantize(out, out);
		
		imshow(fileNames[i], *out);

		char pause = cv::waitKey();  // This introduces a delay between frame.  The length of the delay/number can be reduced.
	}

}

