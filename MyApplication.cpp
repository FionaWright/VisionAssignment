
#include "Utilities.h"
#include <opencv2/opencv.hpp>
#include <cmath>

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

void crop(Mat* src, Mat* dst, cv::Point* srcPoints)
{
	cv::Mat mask = cv::Mat::zeros(src->size(), CV_8UC1);

	cv::fillConvexPoly(mask, srcPoints, 4, cv::Scalar(255));

	*dst = *src;
	dst->setTo(cv::Scalar(0, 0, 0), mask == 0);
}

enum CandidateSource
{
	eHough,
	eContours,
	eWatershed,
};

struct BallCandidate
{
	cv::Point2f Center;
	float Radius;
	CandidateSource Source;  
};

std::vector<BallCandidate> filterCandidates(const std::vector<BallCandidate>& candidates, const cv::Mat& image, double threshold)
{
	std::vector<BallCandidate> filtered;

	cv::Mat circleMask = cv::Mat::zeros(image.size(), CV_8UC1);

	for (const auto& candidate : candidates)
	{
		circleMask.setTo(0);

		cv::Point center(cvRound(candidate.Center.x), cvRound(candidate.Center.y));

		float radius = cvRound(candidate.Radius);

		if (radius <= 0.0f)
			continue;

		cv::circle(
			circleMask,
			center,
			radius,
			cv::Scalar(255),
			cv::FILLED
		);

		double totalPixels = cv::countNonZero(circleMask);

		if (totalPixels == 0)
			continue;

		cv::Mat intersection;
		cv::bitwise_and(image, circleMask, intersection);

		double ratio = static_cast<double>(cv::countNonZero(intersection)) / totalPixels;

		if (ratio >= threshold)
			filtered.push_back(candidate);
	}

	return filtered;
}

void computeFeltMask(Mat* src, Mat* dst)
{
	cv::Mat HLS, H, L, S;
	{
		cv::cvtColor(*src, HLS, cv::COLOR_BGR2HLS);

		std::vector<cv::Mat> channels;
		cv::split(HLS, channels);

		H = channels[0];
		L = channels[1];
		S = channels[2];
	}

	cv::Mat feltMask;
	{
		cv::Mat hsv;
		cv::cvtColor(*src, hsv, cv::COLOR_BGR2HSV);

		cv::inRange(
			hsv,
			cv::Scalar(40, 40, 65),
			cv::Scalar(90, 255, 255),
			feltMask
		);
	}

	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));

	cv::Mat inverseEdges;
	{
		Mat col = HLS;
		//Mat col = H;
		//Mat col = L;
		//col.setTo(0, greenMask);
		cv::GaussianBlur(col, col, cv::Size(7, 7), 1.5);

		cv::Mat edges;
		cv::Canny(col, edges, 15, 20);

		cv::morphologyEx(edges, edges, cv::MORPH_DILATE, kernel);
		cv::morphologyEx(edges, edges, cv::MORPH_DILATE, kernel);
		cv::morphologyEx(edges, edges, cv::MORPH_ERODE, kernel);

		cv::bitwise_not(edges, inverseEdges);

		//imshow(".", edges);
	}

	{
		cv::Mat hsv;
		cv::cvtColor(*src, hsv, cv::COLOR_BGR2HSV);

		hsv.setTo(cv::Scalar(0, 0, 0), inverseEdges == 0);

		cv::Mat floodMask = cv::Mat::zeros(hsv.rows + 2, hsv.cols + 2, CV_8UC1);

		cv::Point seed(600, 400); // TODO: Find stochastically

		float hTol = 5;
		float lTol = 25;
		float vTol = 25;
		cv::Scalar loDiff(hTol, lTol, vTol);
		cv::Scalar upDiff(hTol, lTol, vTol);

		cv::floodFill(
			hsv,
			floodMask,
			seed,
			cv::Scalar(),
			nullptr,
			loDiff,
			upDiff,
			4 | cv::FLOODFILL_MASK_ONLY | (255 << 8)
		);

		feltMask = floodMask(cv::Rect(1, 1, hsv.cols, hsv.rows)).clone();

		//cv::morphologyEx(greenMask, greenMask, cv::MORPH_OPEN, kernel);
		cv::morphologyEx(feltMask, feltMask, cv::MORPH_CLOSE, kernel);

		//imshow(".", greenMask);
	}

	*dst = feltMask;
}

std::vector<BallCandidate> detectBallsHough(Mat* image)
{
	cv::Mat H, L, S;
	{
		cv::Mat hls;
		cv::cvtColor(*image, hls, cv::COLOR_BGR2HLS);

		std::vector<cv::Mat> channels;
		cv::split(hls, channels);

		H = channels[0];
		L = channels[1];
		S = channels[2];
	}

	cv::Mat feltMask;
	computeFeltMask(image, &feltMask);

	cv::Mat gray;
	{
		//cv::cvtColor(*image, gray, cv::COLOR_BGR2GRAY);
		gray = L;

		const int kernelSize = 7;
		const double strength = 1.2;
		cv::GaussianBlur(gray, gray, cv::Size(kernelSize, kernelSize), strength);
		gray.setTo(0, feltMask);
	}

	imshow(".", gray);

	std::vector<BallCandidate> candidates;
	{
		const double inverseAccumReso = 1.2;
		const double minCenterDistance = 10.0;
		const double upperCannyThreshold = 60;
		const double accumThreshold = 12.0;
		const int minRadius = 23;
		const int maxRadius = 26;

		std::vector<cv::Vec3f> circles;

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

		for (int i = 0; i < circles.size(); i++)
		{
			BallCandidate candidate;
			candidate.Center = cv::Point(circles[i][0], circles[i][1]);
			candidate.Radius = circles[i][2];
			candidate.Source = eHough;
			candidates.emplace_back(candidate);
		}
	}

	candidates = filterCandidates(candidates, gray, 0.6f);

	return candidates;
}

std::vector<BallCandidate> combineCircles(
	const std::vector<BallCandidate>& candidates,
	float centerThreshold = 15.0f,
	float radiusThreshold = 8.0f)
{
	std::vector<bool> used(candidates.size(), false);
	std::vector<BallCandidate> result;

	for (size_t i = 0; i < candidates.size(); ++i)
	{
		if (used[i])
			continue;

		float sumX = 0.0f, sumY = 0.0f, sumR = 0.0f;
		int count = 0;

		for (size_t j = i; j < candidates.size(); ++j)
		{
			if (used[j])
				continue;

			float dx = candidates[i].Center.x - candidates[j].Center.x;
			float dy = candidates[i].Center.y - candidates[j].Center.y;

			float d = std::sqrt(dx * dx + dy * dy);
			float dr = std::abs(candidates[i].Radius - candidates[j].Radius);

			if (d <= centerThreshold && dr <= radiusThreshold)
			{
				used[j] = true;
				sumX += candidates[j].Center.x;
				sumY += candidates[j].Center.y;
				sumR += candidates[j].Radius;
				++count;
			}
		}

		result.push_back({
			cv::Point2f(sumX / count, sumY / count),
			sumR / count,
			candidates[i].Source
			});
	}

	return result;
}

// For Debugging
void drawHollowCircles(
	Mat* image,
	const std::vector<BallCandidate>& circles,
	const cv::Scalar& color = cv::Scalar(0, 0, 255),
	int thickness = 2)
{
	for (const auto& circle : circles)
	{
		cv::circle(*image, circle.Center, circle.Radius, color, thickness);
	}
}

enum BallID : uint32_t
{
	eWhite = 0,
	eYellowSolid = 1,
	eBlueSolid = 2,
	eRedSolid = 3,
	ePurpleSolid = 4,
	eOrangeSolid = 5,
	eGreenSolid = 6,
	eBrownSolid = 7,
	eBlack = 8,
	eYellowStriped = 9,
	eBlueStriped = 10,
	eRedStriped = 11,
	ePurpleStriped = 12,
	eOrangeStriped = 13,
	eGreenStriped = 14,
	eBrownStriped = 15,
};

struct AssignedCandidate
{
	BallCandidate Candidate;
	BallID ID;
};

float colorWeight(const cv::Vec3f& observedLab, const cv::Vec3f& expectedLab, float sigma = 15.0f)
{
	cv::Vec3f d = observedLab - expectedLab;
	float dist = std::sqrt(d.dot(d));

	return std::exp(-(dist * dist) / (2.0f * sigma * sigma)); // Gaussian
}

float averageColorWeight(
	const cv::Mat& labImage,
	const BallCandidate& candidate,
	const cv::Vec3f& expectedLab,
	float sigma = 15.0f)
{
	cv::Point center(cvRound(candidate.Center.x), cvRound(candidate.Center.y));
	int radius = cvRound(candidate.Radius);

	cv::Rect bounds(0, 0, labImage.cols, labImage.rows);
	cv::Rect roi(
		center.x - radius,
		center.y - radius,
		radius * 2 + 1,
		radius * 2 + 1);

	roi &= bounds;
	if (roi.empty())
		return 0.0f;

	cv::Mat circleMask = cv::Mat::zeros(roi.size(), CV_8UC1);
	cv::Point localCenter(center.x - roi.x, center.y - roi.y);
	cv::circle(circleMask, localCenter, radius, cv::Scalar(255), cv::FILLED);

	double totalWeight = 0.0;
	int pixelCount = 0;

	for (int y = 0; y < roi.height; ++y)
	{
		for (int x = 0; x < roi.width; ++x)
		{
			if (circleMask.at<uchar>(y, x) == 0)
				continue;

			cv::Vec3b pixel = labImage.at<cv::Vec3b>(roi.y + y, roi.x + x);
			cv::Vec3f pixelLab(pixel[0] * (100.0f / 255.0f), pixel[1] - 128.0f, pixel[2] - 128.0f);

			totalWeight += colorWeight(pixelLab, expectedLab, sigma);
			++pixelCount;
		}
	}

	return pixelCount > 0 ? static_cast<float>(totalWeight / pixelCount) : 0.0f;
}

void printWeightMatrix(const std::vector<std::vector<float>>& weightMatrix)
{
	if (weightMatrix.empty())
	{
		std::cout << "Weight matrix is empty.\n";
		return;
	}

	std::cout << std::setw(12) << "Candidate";
	for (size_t j = 0; j < weightMatrix[0].size(); ++j)
		std::cout << std::setw(8) << j;
	std::cout << '\n';

	std::cout << std::string(12 + weightMatrix[0].size() * 8, '-') << '\n';

	for (size_t i = 0; i < weightMatrix.size(); ++i)
	{
		std::cout << std::setw(12) << i;

		for (float weight : weightMatrix[i])
			std::cout << std::setw(8) << std::fixed << std::setprecision(3) << weight;

		std::cout << '\n';
	}
}

std::vector<AssignedCandidate> buildWeightMatrix(Mat* image, const std::vector<BallCandidate>& candidates)
{
	std::vector<cv::Vec3f> paletteLab;
	{
		// TODO: Extract from training data?
		cv::Vec3b rgbPalette[] = {
			{255, 255, 255}, // White
			{255, 255,   0}, // Yellow
			{  0,   0, 255}, // Blue 
			{255,   0,   0}, // Red
			{128,   0, 128}, // Purple
			{255, 165,   0}, // Orange
			{  0, 128,   0}, // Green
			{165,  42,  42}, // Brown
			{  0,   0,   0}  // Black
		};

		for (const auto& rgb : rgbPalette)
		{
			cv::Mat pixel(1, 1, CV_8UC3);
			pixel.at<cv::Vec3b>(0, 0) = cv::Vec3b(
				rgb[2], rgb[1], rgb[0]
			); // RGB -> BGR

			cv::Mat lab;
			cv::cvtColor(pixel, lab, cv::COLOR_BGR2Lab);

			cv::Vec3b p = lab.at<cv::Vec3b>(0, 0);

			paletteLab.emplace_back(
				p[0] * (100.0f / 255.0f),
				static_cast<float>(p[1]) - 128.0f,
				static_cast<float>(p[2]) - 128.0f
			);
		}
	}

	Mat feltMask;
	computeFeltMask(image, &feltMask);

	cv::Mat lab;
	cv::cvtColor(*image, lab, cv::COLOR_BGR2Lab);

	lab.setTo(cv::Scalar(0,0,0), feltMask);

#define NUM_BALLS 16

	std::vector<std::vector<float>> weightMatrix;

	for (uint32_t i = 0; i < candidates.size(); i++)
	{
		BallCandidate candidate = candidates[i];
		candidate.Radius *= 0.9f; // Exclude edges

		std::vector<float> weights;

		for (uint32_t j = 0; j < NUM_BALLS; j++)
		{
			uint32_t colorIdx = j % 8;
			cv::Vec3f expectedLab = paletteLab[colorIdx];
			
			float weight = averageColorWeight(lab, candidate, expectedLab, 15.0f);

			weights.emplace_back(weight);
		}

		weightMatrix.emplace_back(std::move(weights));
	}

	printWeightMatrix(weightMatrix);

	return {};
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
		cv::Point srcPoints[4] = {
			{337,  35},
			{1570, 21},
			{1888, 451},
			{38, 472},
		};

		size_t width = in->cols;
		size_t height = in->rows;

		crop(in, out, srcPoints);

		std::vector<BallCandidate> circlesH = detectBallsHough(out);

		std::vector<BallCandidate> candidates;
		candidates.insert(candidates.end(), circlesH.begin(), circlesH.end());

		std::vector<BallCandidate> combined = combineCircles(candidates, 21.0f, 8.0f);

		buildWeightMatrix(out, combined);

		drawHollowCircles(out, circlesH, cv::Scalar(0, 0, 255));
		drawHollowCircles(out, combined, cv::Scalar(255, 255, 255));

		//nearestNeighbourQuantize(out, out);
		
		imshow(fileNames[i], *out);

		char pause = cv::waitKey();  // This introduces a delay between frame.  The length of the delay/number can be reduced.
	}

}

/*
* 
* TODO:
* 
* Reject net samples
* Figure out how to get the tucked green ball sample
* Figure out flood fill seed stochastically / chunking / multi-scale?
* Evaluation metrics per image
* Color detection
* Figure out if I'm allowed/required to use the training image for information (Ball colors, table position, nets, etc)
*/
