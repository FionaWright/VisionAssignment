
#include "Utilities.h"
#include <opencv2/opencv.hpp>
#include <cmath>

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
	eCount = 16
};

struct AssignedCandidate
{
	BallCandidate Candidate;
	BallID ID;
};

struct Config
{
	// Input
	const char* FolderPath = "Media/Pool/";
	std::vector<const char*> FileNameList = {
		"AllBalls.jpeg",
		"Table1.jpeg",
		"Table2.jpeg",
		"Table3.jpeg",
		"Table4.jpeg",
		"Table5.jpeg",
		"Table6.jpeg"
	};
	bool RunAllImages = true;

	// Table
	cv::Point TablePoints[4] = { // NW, NE, SE, SW
		{337,  35},
		{1570, 21},
		{1888, 451},
		{38, 472},
	};
	cv::Point2f TablePointsF[4] = { 
		{337,  35},
		{1570, 21},
		{1888, 451},
		{38, 472},
	};

	// Felt Mask
	int FeltMaskMorphKernel = 3;
	int FeltMaskGaussianKernel = 7;
	float FeltMaskGaussianStrength = 1.5f;

	// Canny
	int CannyWeakThreshold = 15;
	int CannyStrongThreshold = 20;
	int CannyNumDilations = 2;
	int CannyNumErosions = 1;

	// Flood Fill
	float FloodFillToleranceHue = 5;
	float FloodFillToleranceLightness = 25;
	float FloodFillToleranceValue = 25;

	// Hough Transform
	int HoughGaussianKernel = 7;
	float HoughGaussianStrength = 1.2f;
	double HoughInverseAccumReso = 1.2;
	double HoughMinCenterDistance = 10.0;
	double HoughUpperCannyThreshold = 60;
	double HoughAccumThreshold = 12.0;
	int HoughMinRadius = 23;
	int HoughMaxRadius = 26;

	// Candidate Filters
	float CandidateFilterNonZeroThreshold = 1.0f; // TODO
	float CandidateCombineDistanceThreshold = 21.0f;
	float CandidateCombineRadiusThreshold = 8.0f;
	float OverlapRejectionThreshold = 1.0f;

	// Recognition
	float LabGaussianStdDev = 1.5f;
	float RecognitionRadiusMult = 0.8f;
	float RecognitionStripesWeight = 0.3f;
	float RecognitionMinWeightThreshold = 0.001f;
	bool UseHungarianSolver = true;
	std::vector<cv::Vec3f> RecognitionPalette = {
		{0.9f,  0.8f,  0.5f},  // 00: White
		{0.9f,  0.5f,  0.0f},  // 01: Yellow
		{0.0f,  0.0f,  0.3f},  // 02: Blue
		{0.9f,  0.0f,  0.0f},  // 03: Red
		{0.05f, 0.0f,  0.1f},  // 04: Purple
		{0.9f,  0.15f, 0.0f},  // 05: Orange
		{0.0f,  0.1f,  0.0f},  // 06: Green
		{0.3f,  0.05f, 0.05f}, // 07: Brown
		{0.0f,  0.0f,  0.0f},  // 08: Black
		{0.9f,  0.5f,  0.0f},  // 09: Yellow stripe
		{0.0f,  0.0f,  0.3f},  // 10: Blue stripe
		{0.9f,  0.0f,  0.0f},  // 11: Red stripe
		{0.05f, 0.0f,  0.1f},  // 12: Purple stripe
		{0.9f,  0.15f, 0.0f},  // 13: Orange stripe
		{0.0f,  0.1f,  0.0f},  // 14: Green stripe
		{0.3f,  0.05f, 0.05f}  // 15: Brown stripe
	};

	// Drawing
	int CircleThickness = 2;
	float EvalCircleDrawRadius = 10.0f;
	std::vector<cv::Vec3b> DrawPalette = {
		{255, 255, 255}, // White
		{  0, 255, 255}, // Yellow
		{255,   0,   0}, // Blue
		{  0,   0, 255}, // Red
		{128,   0, 128}, // Purple
		{  0, 165, 255}, // Orange
		{  0, 128,   0}, // Green
		{ 42,  42, 165}, // Brown
		{  0,   0,   0}, // Black
		{  0, 255, 255}, // Yellow
		{255,   0,   0}, // Blue
		{  0,   0, 255}, // Red
		{128,   0, 128}, // Purple
		{  0, 165, 255}, // Orange
		{  0, 128,   0}, // Green
		{ 42,  42, 165}, // Brown
	};

	// Evaluation
	float MissingBallPenalty = -1.0f;
	float MissingBallReward = 1.0f;
	float EvalDistanceThreshold = 100.0f;
	float EvalDistanceMax = 100.0f;
	float DistancePenaltyWeight = 0.5f;
};
static Config s_Config;

bool BallIdIsStriped(int id) { return id >= 9; }

void loadImage(Mat* image, const std::string& path)
{
	*image = imread(path, -1);

	if (image->empty())
	{
		std::cout << "Could not open " << path << std::endl;
	}
}

void crop(Mat* src, Mat* dst, cv::Point* srcPoints, cv::Scalar value)
{
	cv::Mat mask = cv::Mat::zeros(src->size(), CV_8UC1);

	cv::fillConvexPoly(mask, srcPoints, 4, cv::Scalar(255));

	*dst = src->clone();
	dst->setTo(value, mask == 0);
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

cv::Point2f perspectiveTransform(cv::Point2f point, cv::Point2f* srcPoints, size_t width, size_t height)
{
	// NW, NE, SE, SW
	cv::Point2f dstPoints[4] = {
		{0.0f,          0.0f},
		{width - 1.0f,   0.0f},
		{width - 1.0f,   height - 1.0f},
		{0.0f,           height - 1.0f}
	};

	cv::Mat H = cv::getPerspectiveTransform(srcPoints, dstPoints);

	std::vector<cv::Point2f> pointVec = { point };
	std::vector<cv::Point2f> dst;

	cv::perspectiveTransform(pointVec, dst, H);

	return dst[0];
}

cv::Point2f transformToEvalTable(cv::Point2f point, cv::Point2f* srcPoints, size_t width, size_t height)
{
	point = perspectiveTransform(point, srcPoints, width, height);

	point.x *= 1434.0f / float(width);
	point.y *= 683.0f / float(height);

	return point;
}

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

	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(s_Config.FeltMaskMorphKernel, s_Config.FeltMaskMorphKernel));

	cv::Mat inverseEdges;
	{
		Mat col = HLS;
		//Mat col = H;
		//Mat col = L;
		cv::GaussianBlur(col, col, cv::Size(s_Config.FeltMaskGaussianKernel, s_Config.FeltMaskGaussianKernel), s_Config.FeltMaskGaussianStrength);

		cv::Mat edges;
		cv::Canny(col, edges, s_Config.CannyWeakThreshold, s_Config.CannyStrongThreshold);

		for (int i = 0; i < s_Config.CannyNumDilations; i++)
			cv::morphologyEx(edges, edges, cv::MORPH_DILATE, kernel);

		for (int i = 0; i < s_Config.CannyNumErosions; i++)
			cv::morphologyEx(edges, edges, cv::MORPH_ERODE, kernel);

		cv::bitwise_not(edges, inverseEdges);

		//imshow(".", edges);
	}

	cv::Mat feltMask;
	{
		cv::Mat hsv;
		cv::cvtColor(*src, hsv, cv::COLOR_BGR2HSV);

		hsv.setTo(cv::Scalar(0, 0, 0), inverseEdges == 0);

		cv::Mat floodMask = cv::Mat::zeros(hsv.rows + 2, hsv.cols + 2, CV_8UC1);

		cv::Point seed(600, 400); // TODO: Find stochastically

		cv::Scalar loDiff(s_Config.FloodFillToleranceHue, s_Config.FloodFillToleranceLightness, s_Config.FloodFillToleranceValue);
		cv::Scalar upDiff(s_Config.FloodFillToleranceHue, s_Config.FloodFillToleranceLightness, s_Config.FloodFillToleranceValue);

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

std::vector<BallCandidate> detectBallsHough(Mat* image, Mat* feltMask)
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

	cv::Mat gray;
	{
		//cv::cvtColor(*image, gray, cv::COLOR_BGR2GRAY);
		gray = L;

		cv::GaussianBlur(gray, gray, cv::Size(s_Config.HoughGaussianKernel, s_Config.HoughGaussianKernel), s_Config.HoughGaussianStrength);
		gray.setTo(0, *feltMask);
	}

	imshow(".", gray);

	std::vector<BallCandidate> candidates;
	{
		std::vector<cv::Vec3f> circles;

		cv::HoughCircles(
			gray,
			circles,
			cv::HOUGH_GRADIENT,
			s_Config.HoughInverseAccumReso,
			s_Config.HoughMinCenterDistance,
			s_Config.HoughUpperCannyThreshold,
			s_Config.HoughAccumThreshold,
			s_Config.HoughMinRadius,
			s_Config.HoughMaxRadius
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

float colorWeight(const cv::Vec3f& observedLab, const cv::Vec3f& expectedLab, float sigma = s_Config.LabGaussianStdDev)
{
	cv::Vec3f d = observedLab - expectedLab;
	float dist = std::sqrt(d.dot(d));

	return std::exp(-(dist * dist) / (2.0f * sigma * sigma)); // Gaussian
}

float averageColorWeight(
	const cv::Mat& labImage,
	const cv::Mat& discardMask,
	const BallCandidate& candidate,
	const cv::Vec3f& expectedLab,
	float sigma = s_Config.LabGaussianStdDev)
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

			int imgX = roi.x + x;
			int imgY = roi.y + y;
			if (discardMask.at<uchar>(imgY, imgX) == 1)
				continue;

			cv::Vec3b pixel = labImage.at<cv::Vec3b>(imgY, imgX);
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

std::vector<int> hungarianSolver(const std::vector<std::vector<float>>& weightMatrix)
{
	const int N = static_cast<int>(weightMatrix.size());
	if (N == 0)
		return {};

	const int M = static_cast<int>(weightMatrix[0].size());

	if (M < N)
		throw std::invalid_argument("Hungarian algorithm requires columns >= rows");

	std::vector<double> u(N + 1), v(M + 1);
	std::vector<int> p(M + 1), way(M + 1);

	for (int i = 1; i <= N; ++i)
	{
		p[0] = i;
		int j0 = 0;
		std::vector<double> minv(
			M + 1, std::numeric_limits<double>::infinity());
		std::vector<bool> used(M + 1, false);

		do
		{
			used[j0] = true;
			int i0 = p[j0];
			double delta = std::numeric_limits<double>::infinity();
			int j1 = 0;

			for (int j = 1; j <= M; ++j)
			{
				if (used[j])
					continue;

				double cost = -weightMatrix[i0 - 1][j - 1];
				double cur = cost - u[i0] - v[j];

				if (cur < minv[j])
				{
					minv[j] = cur;
					way[j] = j0;
				}

				if (minv[j] < delta)
				{
					delta = minv[j];
					j1 = j;
				}
			}

			for (int j = 0; j <= M; ++j)
			{
				if (used[j])
				{
					u[p[j]] += delta;
					v[j] -= delta;
				}
				else
				{
					minv[j] -= delta;
				}
			}

			j0 = j1;
		} 
		while (p[j0] != 0);

		do
		{
			int j1 = way[j0];
			p[j0] = p[j1];
			j0 = j1;
		} 
		while (j0 != 0);
	}

	std::vector<int> assignment(N, -1);

	for (int j = 1; j <= M; ++j)
	{
		if (p[j] != 0)
			assignment[p[j] - 1] = j - 1;
	}

	return assignment;
}

std::vector<AssignedCandidate> findBestFits(Mat* image, const std::vector<BallCandidate>& candidates, Mat* feltMask)
{
	std::vector<cv::Vec3f> paletteLab;
	{
		for (const auto& rgb : s_Config.RecognitionPalette)
		{
			cv::Mat pixel(1, 1, CV_8UC3);
			pixel.at<cv::Vec3b>(0, 0) = cv::Vec3b(255 * rgb[2], 255 * rgb[1], 255 * rgb[0]); 

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

	cv::Mat lab;
	cv::cvtColor(*image, lab, cv::COLOR_BGR2Lab);

	std::vector<std::vector<float>> weightMatrix;

	for (uint32_t i = 0; i < candidates.size(); i++)
	{
		BallCandidate candidate = candidates[i];
		candidate.Radius *= s_Config.RecognitionRadiusMult; // Exclude edges

		// Discard ball contributions that overlap closer to the camera
		Mat discardMask = feltMask->clone();
		for (uint32_t other = 0; other < candidates.size(); other++)
		{
			float yDist = candidates[other].Center.y - candidates[i].Center.y;
			if (other == i || yDist < s_Config.OverlapRejectionThreshold)
				continue;

			cv::circle(discardMask, candidates[other].Center, candidates[other].Radius, 255, cv::FILLED);
		}

		std::vector<float> weights;

		for (uint32_t j = 0; j < BallID::eCount; j++)
		{
			cv::Vec3f expectedLab = paletteLab[j];
			
			float weight = averageColorWeight(lab, discardMask, candidate, expectedLab, 15.0f);

			if (BallIdIsStriped(j))
			{
				cv::Vec3f white = paletteLab[0];
				float whiteWeight = averageColorWeight(lab, discardMask, candidate, white, 15.0f);

				weight = whiteWeight * s_Config.RecognitionStripesWeight + weight * (1.0f - s_Config.RecognitionStripesWeight);
			}

			if (weight < s_Config.RecognitionMinWeightThreshold)
				weight = 0.0f;

			weights.emplace_back(weight);
		}

		weightMatrix.emplace_back(std::move(weights));
	}

	printWeightMatrix(weightMatrix);

	std::vector<AssignedCandidate> assignedList;

	// Greedy Approach
	if (!s_Config.UseHungarianSolver)
	{
		auto selectBall = [&](BallID ballId)
		{
			float maxWeight = -1.0f;
			int maxIdx = -1;
			for (uint32_t i = 0; i < candidates.size(); i++)
			{
				std::vector<float>& weights = weightMatrix[i];
				float weight = weights[ballId];

				if (weight > maxWeight)
				{
					maxWeight = weight;
					maxIdx = i;
				}
			}

			if (maxIdx != -1)
			{
				AssignedCandidate assigned;
				assigned.Candidate = candidates[maxIdx];
				assigned.ID = ballId;
				assignedList.emplace_back(assigned);
			}

			for (uint32_t i = 0; i < candidates.size(); i++)
			{
				for (uint32_t j = 0; j < BallID::eCount; j++)
				{
					if (i != maxIdx && j != ballId)
						continue;

					weightMatrix[i][j] = 0.0f;
				}
			}
		};

		selectBall(BallID::eWhite);
		selectBall(BallID::eBlack);

		selectBall(BallID::eYellowSolid);
		selectBall(BallID::eOrangeSolid);
		selectBall(BallID::eRedSolid);
		selectBall(BallID::eBrownSolid);
		selectBall(BallID::ePurpleSolid);
		selectBall(BallID::eBlueSolid);
		selectBall(BallID::eGreenSolid);

		selectBall(BallID::eYellowStriped);
		selectBall(BallID::eOrangeStriped);
		selectBall(BallID::eRedStriped);
		selectBall(BallID::eBrownStriped);
		selectBall(BallID::ePurpleStriped);
		selectBall(BallID::eBlueStriped);
		selectBall(BallID::eGreenStriped);
	}

	// Hungarian Approach
	if (s_Config.UseHungarianSolver)
	{
		// TODO: If this approach is better, cancel out transpose
		{
			std::vector<std::vector<float>> transposed(BallID::eCount, std::vector<float>(candidates.size()));

			for (size_t i = 0; i < candidates.size(); ++i)
			{
				for (int j = 0; j < BallID::eCount; ++j)
				{
					transposed[j][i] = weightMatrix[i][j];
				}
			}

			weightMatrix = transposed;
		}

		// Fill Dummies
		{
			for (uint32_t i = 0; i < BallID::eCount; i++)
			{
				for (uint32_t j = candidates.size(); j < BallID::eCount; j++)
					weightMatrix[i].emplace_back(0.0f);
			}
		}

		auto results = hungarianSolver(weightMatrix);

		for (uint32_t i = 0; i < results.size(); i++)
		{
			if (results[i] >= 0 && results[i] < candidates.size() && weightMatrix[i][results[i]] > 0.0f)
			{
				AssignedCandidate assigned;
				assigned.Candidate = candidates[results[i]];
				assigned.ID = BallID(i);
				assignedList.emplace_back(assigned);
			}
		}
	}

	return assignedList;
}

void drawHollowCircles(
	Mat* image,
	const std::vector<BallCandidate>& circles,
	const cv::Scalar& color = cv::Scalar(0, 0, 255),
	int thickness = s_Config.CircleThickness)
{
	for (const auto& circle : circles)
	{
		cv::circle(*image, circle.Center, circle.Radius, color, thickness);
	}
}

void drawAssignedColors(Mat* image, const std::vector<AssignedCandidate>& assignedList)
{
	for (uint32_t i = 0; i < assignedList.size(); i++)
	{ 
		BallCandidate circle = assignedList[i].Candidate;
		cv::Vec3b color = s_Config.DrawPalette[assignedList[i].ID];
		cv::circle(*image, circle.Center, circle.Radius, color, s_Config.CircleThickness);

		if (BallIdIsStriped(assignedList[i].ID))
		{
			cv::circle(*image, circle.Center, circle.Radius, cv::Scalar(255,255,255), s_Config.CircleThickness-1);
		}
	}
}

struct EvalBallPosition
{
	int Row;
	int Column;
};

struct EvalImageBallPositions
{
	std::string Filename;
	EvalBallPosition Balls[16]; 
};

static const std::vector<EvalImageBallPositions> s_EvalBallPositions = {
	{
		"AllBalls.jpeg",
		{
			{293, 797},
			{253, 908},
			{250, 959},
			{255, 1013},
			{256, 1072},
			{254, 1132},
			{262, 1184},
			{256, 1244},
			{253, 1307},
			{329, 901},
			{330, 961},
			{335, 1013},
			{336, 1067},
			{334, 1130},
			{345, 1187},
			{342, 1245}
		}
	},
	{
		"Table1.jpeg",
		{
			{288, 29},
			{309, 1092},
			{429, 1212},
			{374, 1132}, // 3
			{335, 1050},
			{285, 1211},
			{261, 1170},
			{352, 1173},
			{331, 1132}, // 8
			{240, 1210},
			{310, 1170},
			{407, 1169},
			{281, 1135},
			{374, 1212}, // 13
			{360, 1091},
			{331, 1213}
		}
	},
	{
		"Table2.jpeg",
		{
			{507, 1169},
			{282, 1103},
			{620, 1224},
			{391, 1121},
			{315, 1046},
			{310, 1305},
			{202, 621},
			{361, 1173},
			{331, 1132},
			{177, 725},
			{185, 1269},
			{665, 798},
			{296, 438},
			{53, 1235},
			{362, 1090},
			{399, 1205}
		}
	},
	{
		"Table3.jpeg",
		{
			{434, 29},
			{23, 908},
			{616, 1224},
			{447, 1199},
			{336, 29},
			{305, 1406},
			{210, 621},
			{260, 1294},
			{226, 1252},
			{177, 725},
			{171, 1270},
			{665, 798},
			{249, 351},
			{53, 1233},
			{169, 901},
			{399, 1201}
		}
	},
	{
		"Table4.jpeg",
		{
			{88, 1392},
			{23, 908},
			{546, 1407},
			{444, 1204},
			{-1, -1},
			{305, 1406},
			{210, 621},
			{260, 1294},
			{226, 1252},
			{181, 724},
			{171, 1270},
			{665, 798},
			{249, 351},
			{58, 1231},
			{169, 901},
			{399, 1201}
		}
	},
	{
		"Table5.jpeg",
		{
			{608, 26},
			{535, 23},
			{546, 1407},
			{444, 1204},
			{-1, -1},
			{305, 1406},
			{131, 25},
			{260, 1294},
			{226, 1252},
			{538, 72},
			{171, 1270},
			{665, 798},
			{249, 351},
			{-1, -1},
			{169, 901},
			{399, 1201}
		}
	},
	{
		"Table6.jpeg",
		{
			{629, 947},
			{535, 23},
			{477, 1364},
			{444, 1204},
			{-1, -1},
			{305, 1406},
			{131, 25},
			{260, 1294},
			{226, 1252},
			{538, 72},
			{171, 1270},
			{-1, -1},
			{249, 351},
			{-1, -1},
			{169, 901},
			{399, 1201}
		}
	}
};

float EvaluatePerformance(const EvalImageBallPositions& eval, std::vector<AssignedCandidate>& candidates, size_t width, size_t height)
{
	float totalScore = 0.0f;

	for (int i = 0; i < BallID::eCount; i++)
	{
		BallCandidate* candidate = nullptr;
		for (int j = 0; j < candidates.size(); j++)
		{
			if (candidates[j].ID == i)
			{
				candidate = &candidates[j].Candidate;
				break;
			}
		}

		EvalBallPosition evalPos = eval.Balls[i];

		bool foundCandidate = candidate != nullptr;
		bool ballOnTable = evalPos.Row != -1 && evalPos.Column != -1;

		if (!foundCandidate && !ballOnTable)
		{
			totalScore += s_Config.MissingBallReward;
			continue;
		}

		if (foundCandidate != ballOnTable)
		{
			std::cout << "WARNING: False Positive / Negative" << std::endl;

			totalScore += s_Config.MissingBallPenalty;
			continue;
		}

		float imageCoordX = evalPos.Column;
		float imageCoordY = evalPos.Row;

		cv::Point2f obsPoint(candidate->Center.x, candidate->Center.y);
		obsPoint = transformToEvalTable(obsPoint, s_Config.TablePointsF, width, height);

		float distX = (obsPoint.x - imageCoordX);
		float distY = (obsPoint.y - imageCoordY);
		float lenSqr = distX * distX + distY * distY;

		lenSqr = std::min(s_Config.EvalDistanceMax, lenSqr);
		totalScore += lenSqr * s_Config.DistancePenaltyWeight;
	}

	return totalScore / float(BallID::eCount); // Reduce magnitude
}

void drawEvalCircles(Mat* image, const EvalImageBallPositions& eval)
{
	for (int i = 0; i < BallID::eCount; i++)
	{
		EvalBallPosition evalPos = eval.Balls[i];

		bool ballOnTable = evalPos.Row != -1 && evalPos.Column != -1;

		if (!ballOnTable)
			continue;

		int imageCoordX = evalPos.Column;
		int imageCoordY = evalPos.Row;
		cv::Point p(imageCoordX, imageCoordY);

		cv::Vec3b color = s_Config.DrawPalette[i];
		cv::circle(*image, p, s_Config.EvalCircleDrawRadius, color, s_Config.CircleThickness);

		if (BallIdIsStriped(i))
		{
			cv::circle(*image, p, s_Config.EvalCircleDrawRadius, cv::Scalar(255, 255, 255), s_Config.CircleThickness - 1);
		}
	}
}

void MyApplication()
{
	Mat* inputImages = new Mat[s_Config.FileNameList.size()];
	Mat* outputImages = new Mat[s_Config.FileNameList.size()];

	float totalScore = 0.0f;
	std::vector<float> scores;

	for (int i = 0; i < s_Config.FileNameList.size(); i++)
	{
		Mat* in = &inputImages[i];
		Mat* out = &outputImages[i];

		std::string filePath = std::string(s_Config.FolderPath) + s_Config.FileNameList[i];
		loadImage(in, filePath);

		size_t width = in->cols;
		size_t height = in->rows;

		crop(in, out, s_Config.TablePoints, cv::Scalar(0,0,0));

		cv::Mat feltMask;
		computeFeltMask(out, &feltMask);
		crop(&feltMask, &feltMask, s_Config.TablePoints, cv::Scalar(1,1,1));

		std::vector<BallCandidate> circlesH = detectBallsHough(out, &feltMask);

		std::vector<BallCandidate> candidates;
		candidates.insert(candidates.end(), circlesH.begin(), circlesH.end());
		std::vector<BallCandidate> combined = combineCircles(candidates, s_Config.CandidateCombineDistanceThreshold, s_Config.CandidateCombineRadiusThreshold);

		std::vector<AssignedCandidate> assignedList = findBestFits(out, combined, &feltMask);

		Mat display = out->clone();
		drawHollowCircles(&display, circlesH, cv::Scalar(0, 0, 255));
		imshow("Hough Candidates", display);

		display = out->clone();
		drawHollowCircles(&display, combined, cv::Scalar(255, 255, 255));
		imshow("Combined Candidates", display);

		display = out->clone();
		drawAssignedColors(&display, assignedList);
		imshow("Recognition", display);

		float score = EvaluatePerformance(s_EvalBallPositions[i], assignedList, width, height);
		scores.emplace_back(score);
		totalScore += score;
		std::cout << s_Config.FileNameList[i] << " score: " << score << std::endl;

		display = out->clone();
		perspectiveTransform(&display, &display, s_Config.TablePointsF, display.cols, display.rows);
		cv::resize(display, display, cv::Size(1434, 683));
		drawEvalCircles(&display, s_EvalBallPositions[i]);
		imshow("Eval", display);

		if (!s_Config.RunAllImages)
			cv::waitKey();
	}

	for (int i = 0; i < s_Config.FileNameList.size(); i++)
	{
		float score = scores[i] / float(s_Config.FileNameList.size());
		std::cout << s_Config.FileNameList[i] << " normalized score: " << score << std::endl;
	}

	totalScore /= float(s_Config.FileNameList.size());
	std::cout << "Total Score:		      " << totalScore << std::endl;
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

/*

Best Scores:

AllBalls.jpeg normalized score: 3.079
Table1.jpeg normalized score: 5.650
Table2.jpeg normalized score: 4.079
Table3.jpeg normalized score: 4.867
Table4.jpeg normalized score: 4.443
Table5.jpeg normalized score: 4.463
Table6.jpeg normalized score: 2.670
Total Score:                  29.251

*/