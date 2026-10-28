
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
			size_t minLenSqr = INT_MAX;

			for (size_t i = 0; i < paletteU.size(); ++i)
			{
				size_t db = pixel[0] - paletteU[i][0];
				size_t dg = pixel[1] - paletteU[i][1];
				size_t dr = pixel[2] - paletteU[i][2];

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
			{336,  24},
			{1575, 10},
			{1902, 449},
			{25, 470},
		};

		size_t width = in->cols;
		size_t height = in->rows;

		perspectiveTransform(in, out, srcPoints, width, height);

		nearestNeighbourQuantize(out, out);
		
		imshow(fileNames[i], *out);

		char pause = cv::waitKey();  // This introduces a delay between frame.  The length of the delay/number can be reduced.
	}

}

