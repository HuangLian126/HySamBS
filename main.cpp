#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/stitching/stitcher.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <iostream>
#include <stdio.h>
#include <string>
#include <opencv.hpp>
#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <vector>
#include <fstream> 
#include <cvaux.h>
#include <cxcore.h>
#include "DistanceUtils.h"
#include "RandUtils.h"
#include <iomanip>
#include "LBSP.h"

using namespace std;
using namespace cv;

#define DEFAULT_FRAME_SIZE cv::Size(320,240)

void doudong(Mat SrcImg1, Mat SrcImg2, Mat xx){
	int minHessian = 500;
	SurfFeatureDetector detector(minHessian);
	//SiftFeatureDetector detector;
	//OrbFeatureDetector detector;
	vector<KeyPoint> KeyPoint1, KeyPoint2;
	detector.detect(SrcImg1, KeyPoint1);
	detector.detect(SrcImg2, KeyPoint2);

	SurfDescriptorExtractor extractor;
	Mat Descriptor1, Descriptor2;
	extractor.compute(SrcImg1, KeyPoint1, Descriptor1);
	extractor.compute(SrcImg2, KeyPoint2, Descriptor2);

	FlannBasedMatcher matcher;
	vector<vector<DMatch> > MatchePoints;
	matcher.knnMatch(Descriptor1, Descriptor2, MatchePoints, 2);

	vector<DMatch> GoodMatchePoints;
	for (size_t i = 0; i < MatchePoints.size(); i++){
		if (MatchePoints[i][0].distance < 0.6 * MatchePoints[i][1].distance){
			GoodMatchePoints.push_back(MatchePoints[i][0]);
		}
	}

	size_t pipeijishu = GoodMatchePoints.size();
	if (pipeijishu > 4){
		vector <KeyPoint> RAN_KP1, RAN_KP2;
		for (size_t i = 0; i < GoodMatchePoints.size(); i++){
			RAN_KP1.push_back(KeyPoint1[GoodMatchePoints[i].queryIdx]);
			RAN_KP2.push_back(KeyPoint2[GoodMatchePoints[i].trainIdx]);
		}
		vector <Point2f> p01, p02;
		for (size_t i = 0; i < GoodMatchePoints.size(); i++){
			p01.push_back(RAN_KP1[i].pt);
			p02.push_back(RAN_KP2[i].pt);
		}
		Mat homography;
		vector<unsigned char> inliersMask(p01.size());
		homography = findHomography(p01, p02, CV_RANSAC, 3, inliersMask);
		vector<DMatch> matches_ransac;
		for (size_t i = 0; i < inliersMask.size(); i++){
			if (inliersMask[i]){
				matches_ransac.push_back(GoodMatchePoints[i]);
			}
		}
		int ptCount_good = (int)matches_ransac.size();
		Mat p1(ptCount_good, 2, CV_32F, Scalar(0.0f));
		Mat p2(ptCount_good, 2, CV_32F, Scalar(0.0f));
		Point2f p;
		Point2f pp;
		for (int i = 0; i < ptCount_good; i++){
			p = KeyPoint1[matches_ransac[i].queryIdx].pt;
			p1.at<float>(i, 0) = round(p.x);
			p1.at<float>(i, 1) = round(p.y);
			pp = KeyPoint2[matches_ransac[i].trainIdx].pt;
			p2.at<float>(i, 0) = round(pp.x);
			p2.at<float>(i, 1) = round(pp.y);
		}
		int on_count = 0;
		float pianyi_x = 0.0f;
		float pianyi_y = 0.0f;
		for (int i = 0; i < ptCount_good; i++){
			float p44_x = abs(p2.at<float>(i, 0) - p1.at<float>(i, 0));
			float p44_y = abs(p2.at<float>(i, 1) - p1.at<float>(i, 1));
			if ((p44_x < 30.0000f) && (p44_y < 30.0000f)){
				pianyi_x += p1.at<float>(i, 0) - p2.at<float>(i, 0);
				pianyi_y += p1.at<float>(i, 1) - p2.at<float>(i, 1);
				on_count++;
			}
		}
		if (on_count == 0)
			on_count = 1;
		int result_col = (int)round(pianyi_x / on_count);
		int result_row = (int)round(pianyi_y / on_count);
		xx.at<int>(0, 0) = result_row;
		xx.at<int>(0, 1) = result_col;
	}
	else{
		int ptCount_good = (int)GoodMatchePoints.size();
		Mat p1(ptCount_good, 2, CV_32F, Scalar(0.0f));
		Mat p2(ptCount_good, 2, CV_32F, Scalar(0.0f));
		Point2f p;
		Point2f pp;
		for (int i = 0; i < ptCount_good; i++){
			p = KeyPoint1[GoodMatchePoints[i].queryIdx].pt;
			p1.at<float>(i, 0) = round(p.x);
			p1.at<float>(i, 1) = round(p.y);
			pp = KeyPoint2[GoodMatchePoints[i].trainIdx].pt;
			p2.at<float>(i, 0) = round(pp.x);
			p2.at<float>(i, 1) = round(pp.y);
		}
		int on_count = 0;
		float pianyi_x = 0.0f;
		float pianyi_y = 0.0f;
		for (int i = 0; i < ptCount_good; i++){
			float p44_x = abs(p2.at<float>(i, 0) - p1.at<float>(i, 0));
			float p44_y = abs(p2.at<float>(i, 1) - p1.at<float>(i, 1));
			if ((p44_x < 30.0000f) && (p44_y < 30.0000f)){
				pianyi_x += p1.at<float>(i, 0) - p2.at<float>(i, 0);
				pianyi_y += p1.at<float>(i, 1) - p2.at<float>(i, 1);
				on_count++;
			}
		}
		if (on_count == 0)
			on_count = 1;
		int result_col = (int)round(pianyi_x / on_count);
		int result_row = (int)round(pianyi_y / on_count);
		xx.at<int>(0, 0) = result_row;
		xx.at<int>(0, 1) = result_col;
	}
}

void JiaoZheng(Mat Currents, Mat& Current_JiaoZheng, Mat A, int rows, int cols){
	int PianYi_row_a = A.at<int>(0, 0);
	int PianYi_col_a = A.at<int>(0, 1);
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++){
			if ((i + PianYi_row_a) >= 0 && (j + PianYi_col_a) >= 0 && (i + PianYi_row_a) < rows && (j + PianYi_col_a) < cols){
				uchar r = Currents.at<Vec3b>(i, j)[2];
				uchar g = Currents.at<Vec3b>(i, j)[1];
				uchar b = Currents.at<Vec3b>(i, j)[0];
				Current_JiaoZheng.at<Vec3b>(i + PianYi_row_a, j + PianYi_col_a)[2] = r;
				Current_JiaoZheng.at<Vec3b>(i + PianYi_row_a, j + PianYi_col_a)[1] = g;
				Current_JiaoZheng.at<Vec3b>(i + PianYi_row_a, j + PianYi_col_a)[0] = b;
			}
		}
	}
}

void JiaoZhengGray(Mat Currents, Mat& Current_JiaoZheng, Mat A, int rows, int cols){
	int PianYi_row_a = A.at<int>(0, 0);
	int PianYi_col_a = A.at<int>(0, 1);
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++){
			if ((i + PianYi_row_a) >= 0 && (j + PianYi_col_a) >= 0 && (i + PianYi_row_a) < rows && (j + PianYi_col_a) < cols){
				uchar gray = Currents.at<uchar>(i, j);
				Current_JiaoZheng.at<uchar>(i + PianYi_row_a, j + PianYi_col_a) = gray;
			}
		}
	}
}


void Recover(Mat CurrentMask, Mat& CurrentMaskRecover, Mat A2, int rows, int cols){
	int PianYi_row_b = A2.at<int>(0, 0);
	int PianYi_col_b = A2.at<int>(0, 1);
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++){
			if ((i - PianYi_row_b) >= 0 && (j - PianYi_col_b) >= 0 && (i - PianYi_row_b) < rows && (j - PianYi_col_b) < cols){
				uchar r = CurrentMask.at<uchar>(i, j);
				CurrentMaskRecover.at<uchar>(i - PianYi_row_b, j - PianYi_col_b) = r;
			}
		}
	}
}


void cunzai(Mat& RefBgEdge, Mat& CurrImg, Mat& SegImg, Mat& image, Mat& object_cunzai, int row, int col)
{
	Mat CurrImgEdge = Mat::zeros(image.size(), CV_8UC1);
	Canny(CurrImg, CurrImgEdge, 100, 200, 3);
	Mat SegImgEdge = Mat::zeros(image.size(), CV_8UC1);
	Canny(SegImg, SegImgEdge, 100, 200, 3);
	Mat SegImgDilated = Mat::zeros(image.size(), CV_8UC1);
	dilate(SegImg, SegImgDilated, Mat(), Point(-1, -1), 1);
	Mat SegDilatedCondComt = Mat::zeros(image.size(), CV_8UC1);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(SegImgDilated, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
	for (int i = 0; i<(int)contours.size(); i++){
		Scalar color(255 - i);
		drawContours(SegDilatedCondComt, contours, i, color, CV_FILLED, 8, hierarchy);
	}
	for (int k = 0; k<(int)contours.size(); k++){
		Mat foreground_region = Mat::zeros(image.size(), CV_8UC1);
		Mat background_region = Mat::zeros(image.size(), CV_8UC1);
		Mat current_region = Mat::zeros(image.size(), CV_8UC1);
		for (int i = 0; i < row; i++){
			for (int j = 0; j < col; j++){
				if (SegDilatedCondComt.at<uchar>(i, j) == (255 - k)){
					foreground_region.at<uchar>(i, j) = SegImgEdge.at<uchar>(i, j);
					background_region.at<uchar>(i, j) = RefBgEdge.at<uchar>(i, j);
					current_region.at<uchar>(i, j) = CurrImgEdge.at<uchar>(i, j);
				}
			}
		}
		int z_background_count = 0;
		int z_current_count = 0;
		for (int i = 0; i < row; i++){
			for (int j = 0; j < col; j++){
				if (foreground_region.at<uchar>(i, j) == 255 && current_region.at<uchar>(i, j) == 255)
					z_current_count++;
				if (foreground_region.at<uchar>(i, j) == 255 && background_region.at<uchar>(i, j) == 255)
					z_background_count++;
			}
		}
		if ((z_current_count - z_background_count) > 10){
			for (int i = 0; i < row; i++){
				for (int j = 0; j < col; j++){
					if (SegDilatedCondComt.at<uchar>(i, j) == (255 - k))
						object_cunzai.at<uchar>(i, j) = 255;
				}
			}
		}
		else if ((z_background_count - z_current_count) > 10){
			for (int i = 0; i < row; i++){
				for (int j = 0; j < col; j++){
					if (SegDilatedCondComt.at<uchar>(i, j) == (255 - k))
						object_cunzai.at<uchar>(i, j) = 125;
				}
			}
		}
	}
}


void MyBS(const char* input, const char* OutPathPost, int TotFrames, const int P, const float refs, const char* input_ROI){
	char  fileName[150];
	char fileName2[150];
	char fileName3[150];
	char fileName4[150];
	char fileName5[150];
	char fileName6[150];
	char fileName7[150];
	char fileName8[150];
	char fileName9[150];
	char fileName10[150];

	Mat image;
	Mat image2;
	Mat image3;
	Mat image4;
	Mat image_test;
	sprintf(fileName, input, P);
	image = imread(fileName);

	CV_Assert(!image.empty() && image.cols>0 && image.rows>0);
	CV_Assert(image.isContinuous());
	CV_Assert(image.type() == CV_8UC3 || image.type() == CV_8UC1);

	size_t m_nImgChannels;
	if (image.type() == CV_8UC3) {
		std::vector<cv::Mat> voInitImgChannels;
		cv::split(image, voInitImgChannels);
		if (!cv::countNonZero((voInitImgChannels[0] != voInitImgChannels[1]) | (voInitImgChannels[2] != voInitImgChannels[1]))){
			std::cout << "Warning!!! : grayscale images should always be passed in CV_8UC1 format for optimal performance." << std::endl;
			m_nImgChannels = 1;
		}
		else
			m_nImgChannels = 3;
	}
	else
		m_nImgChannels = 1;

	Mat imageBase;
	if (m_nImgChannels == 3){
		sprintf(fileName10, input, P);
		imageBase = imread(fileName, 1);
	}
	else{
		sprintf(fileName10, input, P);
		imageBase = imread(fileName, 0);
	}


	int m_nDefaultMedianBlurKernelSize = 9;/////////////////////////////////////////////////////////////////
	int m_nMedianBlurKernelSize;
	Mat NewBGROI;
	Mat ROI;
	Mat m_oROI;
	sprintf(fileName, input_ROI, 1);
	ROI = imread(fileName, 0);

	if (ROI.empty() && ROI.size() != image.size()) {
		NewBGROI.create(image.size(), CV_8UC1);
		NewBGROI = cv::Scalar_<uchar>(UCHAR_MAX);
		//	std::cout << "kong" << std::endl;//}//std::cout << "dfffdfd" << std::endl;
	}
	else {
		CV_Assert(ROI.size() == image.size() && ROI.type() == CV_8UC1);
		CV_Assert(countNonZero((ROI<UCHAR_MAX)&(ROI>0)) == 0);
		NewBGROI = ROI.clone();
		cv::Mat oTempROI;
		cv::dilate(NewBGROI, oTempROI, cv::Mat(), cv::Point(-1, -1), LBSP::PATCH_SIZE / 2);
		cv::bitwise_or(NewBGROI, oTempROI / 2, NewBGROI);
	}

	const size_t nOrigROIPxCount = (size_t)cv::countNonZero(NewBGROI);
	CV_Assert(nOrigROIPxCount>0);

	LBSP::validateROI(NewBGROI);
	const size_t nFinalROIPxCount = (size_t)cv::countNonZero(NewBGROI);
	CV_Assert(nFinalROIPxCount>0);

	m_oROI = NewBGROI;
	Size m_oImgSize = image.size();

	size_t m_nTotPxCount = m_oImgSize.area();
	size_t m_nTotRelevantPxCount = nFinalROIPxCount;

	const int nTotImgPixels = m_oImgSize.height*m_oImgSize.width;
	if (nOrigROIPxCount >= m_nTotPxCount / 2 && (int)m_nTotPxCount >= DEFAULT_FRAME_SIZE.area()) {
		const int nRawMedianBlurKernelSize = std::min((int)floor((float)nTotImgPixels / DEFAULT_FRAME_SIZE.area() + 0.5f) + m_nDefaultMedianBlurKernelSize, 14);
		m_nMedianBlurKernelSize = (nRawMedianBlurKernelSize % 2) ? nRawMedianBlurKernelSize : nRawMedianBlurKernelSize - 1;
	}
	else {
		m_nMedianBlurKernelSize = m_nDefaultMedianBlurKernelSize;
	}
	cout << "m_nMedianBlurKernelSize： " << m_nMedianBlurKernelSize << endl;


	bool m_bLearningRateScalingEnabled = true;
	bool m_bAutoModelResetEnabled = true;


	size_t m_nFramesSinceLastReset = 0;
	size_t m_nModelResetCooldown = 0;

	Mat CurrFGMask = Mat::zeros(image.size(), CV_8UC1);
	Mat LastFGMask = Mat::zeros(image.size(), CV_8UC1);
	Mat BlinksFrame(image.size(), CV_8UC1, Scalar(0));
	Mat LastColorFrame(image.size(), CV_8UC((int)m_nImgChannels), Scalar(0));
	Mat LastDescFrame(image.size(), CV_16UC((int)m_nImgChannels), Scalar(0));
	Mat LastRawFGMask(image.size(), CV_8UC1, Scalar(0));
	Mat LastFGMask_dilated(image.size(), CV_8UC1, Scalar(0));
	Mat LastFGMask_dilated_inverted(image.size(), CV_8UC1, Scalar(0));
	Mat CurrFGMask_close(image.size(), CV_8UC1, Scalar(0));
	Mat CurrRawFGBlinkMask(image.size(), CV_8UC1, Scalar(0));
	Mat LastRawFGBlinkMask(image.size(), CV_8UC1, Scalar(0));
	Mat CurrFGMask_close_fill(image.size(), CV_8UC1, Scalar(0));
	Mat CurrDistThresholdFactor(image.size(), CV_32FC1, Scalar(1.0f));
	Mat CurrVariationFactor(image.size(), CV_32FC1, Scalar(10.0f));
	Mat CurrMeanMinDist_LT(image.size(), CV_32FC1, Scalar(0.0f));
	Mat CurrMeanMinDist_ST(image.size(), CV_32FC1, Scalar(0.0f));
	Mat CurrMeanRawSegmRes_LT(image.size(), CV_32FC1, Scalar(0.0f));
	Mat CurrMeanRawSegmRes_ST(image.size(), CV_32FC1, Scalar(0.0f));
	Mat CurrMeanFinalSegmRes_LT(image.size(), CV_32FC1, Scalar(0.0f));
	Mat CurrMeanFinalSegmRes_ST(image.size(), CV_32FC1, Scalar(0.0f));
	Mat CurrMeanLastDist(image.size(), CV_32FC1, Scalar(0.0f));
	Mat UnstableRegionMask(image.size(), CV_8UC1, Scalar(0));
	Mat UpdateRateFrame(image.size(), CV_32FC1, Scalar(8.0000f));
	int row = image.rows;
	int col = image.cols;
	Size ImgSize = image.size();
	Size m_oDownSampledFrameSize = Size(ImgSize.width / 8, ImgSize.height / 8);
	Mat m_oMeanDownSampledLastDistFrame_LT;
	m_oMeanDownSampledLastDistFrame_LT.create(m_oDownSampledFrameSize, CV_32FC((int)m_nImgChannels));
	m_oMeanDownSampledLastDistFrame_LT = cv::Scalar(0.0f);

	Mat m_oMeanDownSampledLastDistFrame_ST;
	m_oMeanDownSampledLastDistFrame_ST.create(m_oDownSampledFrameSize, CV_32FC((int)m_nImgChannels));
	m_oMeanDownSampledLastDistFrame_ST = cv::Scalar(0.0f);

	Mat m_oDownSampledFrame_MotionAnalysis;
	m_oDownSampledFrame_MotionAnalysis.create(m_oDownSampledFrameSize, CV_8UC((int)m_nImgChannels));
	m_oDownSampledFrame_MotionAnalysis = cv::Scalar_<uchar>::all(0);

	//size_t m_nTotPxCount = ImgSize.area();
	size_t* m_aPxIdxLUT;
	struct PxInfoBase{
		int nImgCoord_Y;
		int nImgCoord_X;
		size_t nModelIdx;
	};
	PxInfoBase* m_aPxInfoLUT;
	m_aPxIdxLUT = new size_t[m_nTotPxCount];
	m_aPxInfoLUT = new PxInfoBase[m_nTotPxCount];

	Mat LBSP = Mat::zeros(image.size(), CV_16UC((int)m_nImgChannels));

	float m_fLastNonZeroDescRatio = 0.0f;
	size_t m_anLBSPThreshold_8bitLUT[256];
	if (m_nImgChannels == 3){
		for (size_t t = 0; t <= 255; ++t){
			m_anLBSPThreshold_8bitLUT[t] = saturate_cast<uchar>(t*refs);
		}
	}
	else{  //m_nImgChannels == 1
		for (size_t t = 0; t <= 255; ++t){
			m_anLBSPThreshold_8bitLUT[t] = saturate_cast<uchar>((t*refs) / 3);
		}
	}
	for (size_t nPxIter = 0, nModelIter = 0; nPxIter < m_nTotPxCount; ++nPxIter) {
		m_aPxIdxLUT[nModelIter] = nPxIter;
		m_aPxInfoLUT[nPxIter].nImgCoord_Y = (int)nPxIter / ImgSize.width;
		m_aPxInfoLUT[nPxIter].nImgCoord_X = (int)nPxIter%ImgSize.width;
		m_aPxInfoLUT[nPxIter].nModelIdx = nModelIter;
		++nModelIter;
	}
	
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	const size_t NUMBER = 30;
	Mat BGColorSamples[NUMBER];
	Mat BGColorSamplesR[NUMBER];
	Mat BGColorSamplesG[NUMBER];
	Mat BGColorSamplesB[NUMBER];
	Mat BackColorSamplesR[NUMBER];
	Mat BackColorSamplesG[NUMBER];
	Mat BackColorSamplesB[NUMBER];
	Mat BGCount[NUMBER];
	Mat BGDescSamples[NUMBER];

	Mat BGColorSamplesGray[NUMBER];
	Mat BackColorSamplesGray[NUMBER];

	Mat XX(1, 2, CV_32SC1, Scalar(0));
	Mat RefBgEdge = Mat::zeros(image.size(), CV_8UC1);
	Mat RefBG = Mat::zeros(image.size(), CV_8UC((int)m_nImgChannels));

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	if (m_nImgChannels == 3){
		for (int k = 0; k < NUMBER; k++){
			Mat color_gray = Mat::zeros(image.size(), CV_8UC1);
			Mat D = Mat::zeros(image.size(), CV_8UC1);
			BGColorSamplesR[k] = color_gray.clone();
			BGColorSamplesG[k] = color_gray.clone();
			BGColorSamplesB[k] = color_gray.clone();
			BackColorSamplesR[k] = color_gray.clone();
			BackColorSamplesG[k] = color_gray.clone();
			BackColorSamplesB[k] = color_gray.clone();
			BGCount[k] = D.clone();
		}
		for (int k = 1; k <= P * NUMBER; k++){
			if ((k % P) == 0){
				sprintf(fileName2, input, k);
				image2 = imread(fileName2);
				//Mat srcImg22 = Mat::zeros(image.size(), CV_8UC1);
				//cvtColor(image2, srcImg22, CV_BGR2GRAY);
				doudong(imageBase, image2, XX);
				Mat image_JiaoZheng = Mat::zeros(image.size(), CV_8UC3);
				JiaoZheng(image2, image_JiaoZheng, XX, row, col);
				for (int i = 0; i < row; i++){
					for (int j = 0; j < col; j++){
						BGColorSamplesR[k / P - 1].at<uchar>(i, j) = image_JiaoZheng.at<Vec3b>(i, j)[2];
						BGColorSamplesG[k / P - 1].at<uchar>(i, j) = image_JiaoZheng.at<Vec3b>(i, j)[1];
						BGColorSamplesB[k / P - 1].at<uchar>(i, j) = image_JiaoZheng.at<Vec3b>(i, j)[0];
					}
				}
			}
		}
		for (int k = 0; k < NUMBER; k++){
			for (int i = 0; i < row; i++){
				for (int j = 0; j < col; j++){
					if (k == 0){
						BackColorSamplesR[k].at<uchar>(i, j) = BGColorSamplesR[k].at<uchar>(i, j);
						BackColorSamplesG[k].at<uchar>(i, j) = BGColorSamplesG[k].at<uchar>(i, j);
						BackColorSamplesB[k].at<uchar>(i, j) = BGColorSamplesB[k].at<uchar>(i, j);
						BGCount[k].at<uchar>(i, j) = (uchar)1;
					}
					else{
						uchar r = BGColorSamplesR[k].at<uchar>(i, j);
						uchar g = BGColorSamplesG[k].at<uchar>(i, j);
						uchar b = BGColorSamplesB[k].at<uchar>(i, j);
						for (int z = 0; z < k; z++){
							uchar B1 = BGColorSamplesR[z].at<uchar>(i, j);
							uchar B2 = BGColorSamplesG[z].at<uchar>(i, j);
							uchar B3 = BGColorSamplesB[z].at<uchar>(i, j);
							uchar B4 = BGCount[z].at<uchar>(i, j);
							float dist_color = (float)sqrt((r - B1)*(r - B1) + (g - B2)*(g - B2) + (b - B3)*(b - B3));
							if (dist_color <= 5.0f){
								BackColorSamplesR[z].at<uchar>(i, j) = (uchar)round((B1*B4 + r) / (B4 + 1));
								BackColorSamplesG[z].at<uchar>(i, j) = (uchar)round((B2*B4 + g) / (B4 + 1));
								BackColorSamplesB[z].at<uchar>(i, j) = (uchar)round((B3*B4 + b) / (B4 + 1));
								BGCount[z].at<uchar>(i, j) = (uchar)(B4 + 1);
								break;
							}
							else{
								BackColorSamplesR[k].at<uchar>(i, j) = (uchar)r;
								BackColorSamplesG[k].at<uchar>(i, j) = (uchar)g;
								BackColorSamplesB[k].at<uchar>(i, j) = (uchar)b;
								BGCount[k].at<uchar>(i, j) = (uchar)1;
							}
						}
					}
				}
			}
		}
		for (int i = 0; i < row; i++){
			for (int j = 0; j < col; j++){
				uchar ss[NUMBER];
				for (int k = 0; k < NUMBER; k++){
					ss[k] = BGCount[k].at<uchar>(i, j);
				}
				uchar max = ss[0];
				int MaxIndex = 0;
				for (int X = 1; X < NUMBER; X++){
					if (ss[X] > max){
						MaxIndex = X;
						max = ss[X];
					}
				}
				RefBG.at<Vec3b>(i, j)[2] = BackColorSamplesR[MaxIndex].at<uchar>(i, j);
				RefBG.at<Vec3b>(i, j)[1] = BackColorSamplesG[MaxIndex].at<uchar>(i, j);
				RefBG.at<Vec3b>(i, j)[0] = BackColorSamplesB[MaxIndex].at<uchar>(i, j);
			}
		}
		//imshow("参考背景", RefBG);
		cvtColor(RefBG, RefBG, CV_BGR2GRAY);
		Canny(RefBG, RefBgEdge, 450, 900, 3);
	}

	else{  //m_nImgChannels == 1
		for (int k = 0; k < NUMBER; k++){
			Mat color_gray = Mat::zeros(image.size(), CV_8UC1);
			Mat D = Mat::zeros(image.size(), CV_8UC1);
			BGColorSamplesGray[k] = color_gray.clone();
			BackColorSamplesGray[k] = color_gray.clone();
			BGCount[k] = D.clone();
		}
		for (int k = 1; k <= P * NUMBER; k++){
			if ((k % P) == 0){
				sprintf(fileName2, input, k);
				image2 = imread(fileName2, 0);
				//Mat srcImg22 = Mat::zeros(image.size(), CV_8UC1);
				//srcImg22 = image2;
				doudong(imageBase, image2, XX);

				Mat image_JiaoZheng = Mat::zeros(image.size(), CV_8UC1);
				JiaoZhengGray(image2, image_JiaoZheng, XX, row, col);
				for (int i = 0; i < row; i++){
					for (int j = 0; j < col; j++){
						BGColorSamplesGray[k / P - 1].at<uchar>(i, j) = image_JiaoZheng.at<uchar>(i, j);
					}
				}
			}
		}
		for (int k = 0; k < NUMBER; k++){
			for (int i = 0; i < row; i++){
				for (int j = 0; j < col; j++){
					if (k == 0){
						BackColorSamplesGray[k].at<uchar>(i, j) = BGColorSamplesGray[k].at<uchar>(i, j);
						BGCount[k].at<uchar>(i, j) = (uchar)1;
					}
					else{
						uchar gray = BGColorSamplesGray[k].at<uchar>(i, j);

						for (int z = 0; z < k; z++){
							uchar gray2 = BGColorSamplesGray[z].at<uchar>(i, j);
							uchar B4 = BGCount[z].at<uchar>(i, j);
							float dist_color = (float)abs(gray - gray2);
							if (dist_color <= 5.0f){
								BackColorSamplesGray[z].at<uchar>(i, j) = (uchar)round((gray2*B4 + gray) / (B4 + 1));
								BGCount[z].at<uchar>(i, j) = (uchar)(B4 + 1);
								break;
							}
							else{
								BackColorSamplesGray[k].at<uchar>(i, j) = (uchar)gray;
								BGCount[k].at<uchar>(i, j) = (uchar)1;
							}
						}
					}
				}
			}
		}
		for (int i = 0; i < row; i++){
			for (int j = 0; j < col; j++){
				uchar ss[NUMBER];
				for (int k = 0; k < NUMBER; k++){
					ss[k] = BGCount[k].at<uchar>(i, j);
				}
				uchar max = ss[0];
				int MaxIndex = 0;
				for (int X = 1; X < NUMBER; X++){
					if (ss[X] > max){
						MaxIndex = X;
						max = ss[X];
					}
				}
				RefBG.at<uchar>(i, j) = BackColorSamplesGray[MaxIndex].at<uchar>(i, j);
			}
		}
		//imshow("参考背景", RefBG);
		Canny(RefBG, RefBgEdge, 450, 900, 3);
	}

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	if (m_nImgChannels == 3){
		for (int k = 1; k <= NUMBER * P; k++){
			if ((k % P) == 0){
				sprintf(fileName3, input, k);
				image3 = imread(fileName3);
				//Mat srcImg33 = Mat::zeros(image.size(), CV_8UC1);
				//cvtColor(image3, srcImg33, CV_BGR2GRAY);

				doudong(image, image3, XX);
				Mat image_JiaoZheng3 = Mat::zeros(image.size(), CV_8UC3);
				JiaoZheng(image3, image_JiaoZheng3, XX, row, col);

				BGColorSamples[k / P - 1] = image_JiaoZheng3.clone();
				for (size_t ModelIter = 0; ModelIter < m_nTotPxCount; ++ModelIter) {
					const size_t PxIter = m_aPxIdxLUT[ModelIter];
					const int nCurrImgCoord_X = m_aPxInfoLUT[PxIter].nImgCoord_X;
					const int nCurrImgCoord_Y = m_aPxInfoLUT[PxIter].nImgCoord_Y;
					if (nCurrImgCoord_X > 1 && nCurrImgCoord_X < col - 2 && nCurrImgCoord_Y>1 && nCurrImgCoord_Y < row - 2){
						const size_t PxIterRGB = PxIter * 3;
						const size_t DescIterRGB = PxIterRGB * 2;
						for (size_t c = 0; c < 3; ++c) {
							LBSP::computeSingleRGBDescriptor(image_JiaoZheng3, image_JiaoZheng3.data[PxIterRGB + c], nCurrImgCoord_X, nCurrImgCoord_Y, c, m_anLBSPThreshold_8bitLUT[image_JiaoZheng3.data[PxIterRGB + c]], ((ushort*)(LBSP.data + DescIterRGB))[c]);
						}
					}
				}
				BGDescSamples[k / P - 1] = LBSP.clone();
			}
		}
	}
	else{  // m_nImgChannels == 1
		for (int k = 1; k <= NUMBER * P; k++){
			if ((k % P) == 0){
				sprintf(fileName3, input, k);
				image3 = imread(fileName3, 0);
				//Mat srcImg33 = Mat::zeros(image.size(), CV_8UC1);
				//cvtColor(image3, srcImg33, CV_BGR2GRAY);
				doudong(imageBase, image3, XX);

				Mat image_JiaoZheng3 = Mat::zeros(image.size(), CV_8UC1);
				JiaoZhengGray(image3, image_JiaoZheng3, XX, row, col);
				BGColorSamples[k / P - 1] = image_JiaoZheng3.clone();
				for (size_t ModelIter = 0; ModelIter < m_nTotPxCount; ++ModelIter) {
					const size_t PxIter = m_aPxIdxLUT[ModelIter];
					const int nCurrImgCoord_X = m_aPxInfoLUT[PxIter].nImgCoord_X;
					const int nCurrImgCoord_Y = m_aPxInfoLUT[PxIter].nImgCoord_Y;
					if (nCurrImgCoord_X > 1 && nCurrImgCoord_X < col - 2 && nCurrImgCoord_Y>1 && nCurrImgCoord_Y < row - 2){
						const size_t DescIter = PxIter * 2;
						LBSP::computeGrayscaleDescriptor(image_JiaoZheng3, image_JiaoZheng3.data[PxIter], nCurrImgCoord_X, nCurrImgCoord_Y, m_anLBSPThreshold_8bitLUT[image_JiaoZheng3.data[PxIter]], *((ushort*)(LBSP.data + DescIter)));
					}
				}
				BGDescSamples[k / P - 1] = LBSP.clone();
			}
		}
	}


	//////////////////////////////////////////////////////////////////////////////////
	int FrameIndex = 0;
	Mat COM_is_beijing = Mat::zeros(image.size(), CV_16UC1);
	Mat COM_is_qianjing = Mat::zeros(image.size(), CV_16UC1);
	Mat COM_foreground = Mat::zeros(image.size(), CV_16UC1);

	for (int k = P * NUMBER + 1; k <= TotFrames; k++){
		if (m_nImgChannels == 3){
			RNG rng;
			sprintf(fileName4, input, k);
			image4 = imread(fileName4);
			//Mat srcImg44 = Mat::zeros(image.size(), CV_8UC1);
			//cvtColor(image4, srcImg44, CV_BGR2GRAY);
			doudong(imageBase, image4, XX);
			//cout << XX << endl;
			Mat CurrImg = Mat::zeros(image.size(), CV_8UC3);
			JiaoZheng(image4, CurrImg, XX, row, col);
			size_t nNonZeroDescCount = 0;
			const float fRollAvgFactor_LT = 1.0f / min(++FrameIndex, 100);
			const float fRollAvgFactor_ST = 1.0f / min(FrameIndex, 25);
			for (size_t nModelIter = 0; nModelIter < m_nTotPxCount; ++nModelIter){
				const size_t nPxIter = m_aPxIdxLUT[nModelIter];
				const int nCurrImgCoord_X = m_aPxInfoLUT[nPxIter].nImgCoord_X;
				const int nCurrImgCoord_Y = m_aPxInfoLUT[nPxIter].nImgCoord_Y;
				if ((nCurrImgCoord_X > 1) && (nCurrImgCoord_X < col - 2) && (nCurrImgCoord_Y>1) && (nCurrImgCoord_Y < row - 2)){
					const size_t nPxIterRGB = nPxIter * 3;
					const size_t nDescIterRGB = nPxIterRGB * 2;
					const size_t nFloatIter = nPxIter * 4;
					const uchar* const anCurrColor = CurrImg.data + nPxIterRGB;
					float *pfCurrVariationFactor = (float*)(CurrVariationFactor.data + nFloatIter);
					float* pfCurrDistThresholdFactor = (float*)(CurrDistThresholdFactor.data + nFloatIter);
					float* pfCurrLearningRate = ((float*)(UpdateRateFrame.data + nFloatIter));
					float* pfCurrMeanLastDist = ((float*)(CurrMeanLastDist.data + nFloatIter));
					float* pfCurrMeanMinDist_LT = ((float*)(CurrMeanMinDist_LT.data + nFloatIter));
					float* pfCurrMeanMinDist_ST = ((float*)(CurrMeanMinDist_ST.data + nFloatIter));
					float* pfCurrMeanRawSegmRes_LT = ((float*)(CurrMeanRawSegmRes_LT.data + nFloatIter));
					float* pfCurrMeanRawSegmRes_ST = ((float*)(CurrMeanRawSegmRes_ST.data + nFloatIter));
					float* pfCurrMeanFinalSegmRes_LT = ((float*)(CurrMeanFinalSegmRes_LT.data + nFloatIter));
					float* pfCurrMeanFinalSegmRes_ST = ((float*)(CurrMeanFinalSegmRes_ST.data + nFloatIter));
					ushort* LastIntraDesc = ((ushort*)(LastDescFrame.data + nDescIterRGB));
					uchar* LastColor = LastColorFrame.data + nPxIterRGB;
					size_t nCurrColorDistThreshold = (size_t)(*pfCurrDistThresholdFactor*30.0f);
					if (nCurrColorDistThreshold > 70)
						nCurrColorDistThreshold = 70;
					//size_t nCurrDescDistThreshold = (((size_t)1 << ((size_t)floor(*pfCurrDistThresholdFactor + 0.5f)))) * 3;
					size_t nCurrDescDistThreshold = (size_t)round((pow(2, *pfCurrDistThresholdFactor) + 3) * 3);
					if (nCurrDescDistThreshold > 21)
						nCurrDescDistThreshold = 21;
					ushort anCurrInterDesc[3], anCurrIntraDesc[3];
					const size_t anCurrIntraLBSPThresholds[3] = { m_anLBSPThreshold_8bitLUT[anCurrColor[0]], m_anLBSPThreshold_8bitLUT[anCurrColor[1]], m_anLBSPThreshold_8bitLUT[anCurrColor[2]] };
					LBSP::computeRGBDescriptor(CurrImg, anCurrColor, nCurrImgCoord_X, nCurrImgCoord_Y, anCurrIntraLBSPThresholds, anCurrIntraDesc);
					size_t nMinTotDescDist = 48;
					size_t nMinTotSumDist = 255;
					UnstableRegionMask.data[nPxIter] = ((*pfCurrDistThresholdFactor) > 2.000f || (*pfCurrMeanRawSegmRes_LT - *pfCurrMeanFinalSegmRes_LT) > 0.100f || (*pfCurrMeanRawSegmRes_ST - *pfCurrMeanFinalSegmRes_ST) > 0.100f) ? 1 : 0;
					size_t GoodSamplesCount = 0, SampleIdx = 0;
					while (GoodSamplesCount < 2 && SampleIdx < NUMBER){
						const uchar* const anBGColor = BGColorSamples[SampleIdx].data + nPxIterRGB;
						const ushort* const anBGIntraDesc = (ushort*)(BGDescSamples[SampleIdx].data + nDescIterRGB);
						size_t ColorDist = (size_t)(sqrt((anBGColor[0] - anCurrColor[0])*(anBGColor[0] - anCurrColor[0]) + (anBGColor[1] - anCurrColor[1])*(anBGColor[1] - anCurrColor[1]) + (anBGColor[2] - anCurrColor[2])*(anBGColor[2] - anCurrColor[2])));
						if (ColorDist > nCurrColorDistThreshold)
							goto failedcheck3ch;
						size_t LbspDist_Intra = hdist(anCurrIntraDesc[0], anBGIntraDesc[0]) + hdist(anCurrIntraDesc[1], anBGIntraDesc[1]) + hdist(anCurrIntraDesc[2], anBGIntraDesc[2]);
						const size_t anCurrInterLBSPThresholds[3] = { m_anLBSPThreshold_8bitLUT[anBGColor[0]], m_anLBSPThreshold_8bitLUT[anBGColor[1]], m_anLBSPThreshold_8bitLUT[anBGColor[2]] };
						LBSP::computeRGBDescriptor(CurrImg, anBGColor, nCurrImgCoord_X, nCurrImgCoord_Y, anCurrInterLBSPThresholds, anCurrInterDesc);
						size_t LbspDist_Inter = hdist(anCurrInterDesc[0], anBGIntraDesc[0]) + hdist(anCurrInterDesc[1], anBGIntraDesc[1]) + hdist(anCurrInterDesc[2], anBGIntraDesc[2]);
						//size_t LbspDist = (size_t)(LbspDist_Inter + LbspDist_Intra) / 2;
						size_t LbspDist = LbspDist_Inter;
						if (LbspDist > nCurrDescDistThreshold)
							goto failedcheck3ch;
						if (nMinTotDescDist > ColorDist)
							nMinTotDescDist = ColorDist;
						if (nMinTotSumDist > LbspDist)
							nMinTotSumDist = LbspDist;
						GoodSamplesCount++;
					failedcheck3ch:
						SampleIdx++;
					}
					if (GoodSamplesCount < 2){
						const float fNormalizedMinDist = min(1.0f, ((float)nMinTotSumDist / 255 + (float)nMinTotDescDist / 48) / 2 + (float)((2 - GoodSamplesCount) / 2));
						*pfCurrMeanMinDist_LT = (*pfCurrMeanMinDist_LT)*(1.0f - fRollAvgFactor_LT) + fNormalizedMinDist*fRollAvgFactor_LT;
						*pfCurrMeanMinDist_ST = (*pfCurrMeanMinDist_ST)*(1.0f - fRollAvgFactor_ST) + fNormalizedMinDist*fRollAvgFactor_ST;
						*pfCurrMeanRawSegmRes_LT = (*pfCurrMeanRawSegmRes_LT)*(1.0f - fRollAvgFactor_LT) + fRollAvgFactor_LT;
						*pfCurrMeanRawSegmRes_ST = (*pfCurrMeanRawSegmRes_ST)*(1.0f - fRollAvgFactor_ST) + fRollAvgFactor_ST;
						CurrFGMask.data[nPxIter] = UCHAR_MAX;
					}
					else{
						const float fNormalizedMinDist = ((float)(nMinTotSumDist / 255) + (float)(nMinTotDescDist / 48)) / 2;
						*pfCurrMeanMinDist_LT = (*pfCurrMeanMinDist_LT)*(1.0f - fRollAvgFactor_LT) + fNormalizedMinDist*fRollAvgFactor_LT;
						*pfCurrMeanMinDist_ST = (*pfCurrMeanMinDist_ST)*(1.0f - fRollAvgFactor_ST) + fNormalizedMinDist*fRollAvgFactor_ST;
						*pfCurrMeanRawSegmRes_LT = (*pfCurrMeanRawSegmRes_LT)*(1.0f - fRollAvgFactor_LT);
						*pfCurrMeanRawSegmRes_ST = (*pfCurrMeanRawSegmRes_ST)*(1.0f - fRollAvgFactor_ST);
						CurrFGMask.data[nPxIter] = 0;
						size_t Tx = (size_t)round(*pfCurrLearningRate);
						size_t random = rng.uniform(0, Tx);
						if (random == 1){
							random = rng.uniform(0, NUMBER);
							for (size_t c = 0; c < 3; ++c){
								*(BGColorSamples[random].data + nPxIterRGB + c) = anCurrColor[c];
								*((ushort*)(BGDescSamples[random].data + nDescIterRGB + 2 * c)) = anCurrIntraDesc[c];
							}
						}
					}
					if (LastFGMask.data[nPxIter] || (min(*pfCurrMeanMinDist_LT, *pfCurrMeanMinDist_ST) < 0.100f && CurrFGMask.data[nPxIter])){
						if (*pfCurrLearningRate<20.0f)
							*pfCurrLearningRate += 0.5000f / (max(*pfCurrMeanMinDist_LT, *pfCurrMeanMinDist_ST)*(*pfCurrVariationFactor));
					}
					else if (*pfCurrLearningRate>2.0000f)
						*pfCurrLearningRate -= 0.2500f*(*pfCurrVariationFactor) / max(*pfCurrMeanMinDist_LT, *pfCurrMeanMinDist_ST);
					if (*pfCurrLearningRate<2.0000f)
						*pfCurrLearningRate = 2.0000f;
					else if (*pfCurrLearningRate>20.00f)
						*pfCurrLearningRate = 20.00f;
					//if (max(*pfCurrMeanMinDist_LT, *pfCurrMeanMinDist_ST)>0.100f && BlinksFrame.data[nPxIter])
					if (BlinksFrame.data[nPxIter])
						(*pfCurrVariationFactor) += 1.0000f;
					else if ((*pfCurrVariationFactor) > 0.100f){
						*pfCurrVariationFactor -= LastFGMask.data[nPxIter] ? 0.025f : UnstableRegionMask.data[nPxIter] ? 0.050f : 0.100f;
						if ((*pfCurrVariationFactor) < 0.100f)
							(*pfCurrVariationFactor) = 0.100f;
					}
					if ((*pfCurrDistThresholdFactor) < pow(1.0f + min(*pfCurrMeanMinDist_LT, *pfCurrMeanMinDist_ST) * 2, 2)){
						(*pfCurrDistThresholdFactor) += 0.01f*((*pfCurrVariationFactor) - 0.100f);
					}
					else{
						(*pfCurrDistThresholdFactor) -= 0.01f / (*pfCurrVariationFactor);
						if ((*pfCurrDistThresholdFactor) < 1.0f)
							(*pfCurrDistThresholdFactor) = 1.0f;
					}
					if (popcount<3>(anCurrIntraDesc) >= 4)
						++nNonZeroDescCount;
					for (size_t c = 0; c < 3; ++c) {
						LastIntraDesc[c] = anCurrIntraDesc[c];
						LastColor[c] = anCurrColor[c];
					}
				}
			}
			medianBlur(CurrFGMask, CurrFGMask, 3);
			medianBlur(LastRawFGMask, LastRawFGMask, 3);
			bitwise_xor(CurrFGMask, LastRawFGMask, CurrRawFGBlinkMask);
			bitwise_or(CurrRawFGBlinkMask, LastRawFGBlinkMask, BlinksFrame);
			CurrRawFGBlinkMask.copyTo(LastRawFGBlinkMask);
			CurrFGMask.copyTo(LastRawFGMask);
			morphologyEx(CurrFGMask, CurrFGMask_close, MORPH_CLOSE, Mat());
			CurrFGMask_close.copyTo(CurrFGMask_close_fill);
			floodFill(CurrFGMask_close_fill, Point(0, 0), UCHAR_MAX);
			bitwise_not(CurrFGMask_close_fill, CurrFGMask_close_fill);
			erode(CurrFGMask_close, CurrFGMask_close, Mat(), Point(-1, -1), 3);
			bitwise_or(CurrFGMask, CurrFGMask_close_fill, CurrFGMask);
			bitwise_or(CurrFGMask, CurrFGMask_close, CurrFGMask);
			medianBlur(CurrFGMask, LastFGMask, m_nMedianBlurKernelSize);
			dilate(LastFGMask, LastFGMask_dilated, Mat(), Point(-1, -1), 3);
			bitwise_and(BlinksFrame, LastFGMask_dilated_inverted, BlinksFrame);
			bitwise_not(LastFGMask_dilated, LastFGMask_dilated_inverted);
			bitwise_and(BlinksFrame, LastFGMask_dilated_inverted, BlinksFrame);
			LastFGMask.copyTo(CurrFGMask);
			Mat object_cunzai = Mat::zeros(image.size(), CV_8UC1);
			cunzai(RefBgEdge, CurrImg, CurrFGMask, image, object_cunzai, row, col);
			for (size_t nModelIter = 0; nModelIter < m_nTotPxCount; ++nModelIter){
				const size_t nPxIter = m_aPxIdxLUT[nModelIter];
				const size_t nDescIter = nPxIter * 2;
				ushort* counts_COM_foreground = (ushort*)(COM_foreground.data + nDescIter);
				if (CurrFGMask.data[nPxIter] == 255)
					* counts_COM_foreground += 1;
				else
					* counts_COM_foreground = 0;
			}
			for (size_t nModelIter = 0; nModelIter < m_nTotPxCount; ++nModelIter){
				const size_t nPxIter = m_aPxIdxLUT[nModelIter];
				const size_t nDescIter = nPxIter * 2;
				ushort* counts_COM_is_beijing = (ushort*)(COM_is_beijing.data + nDescIter);
				ushort* counts_COM_is_qianjing = (ushort*)(COM_is_qianjing.data + nDescIter);
				if (object_cunzai.data[nPxIter] == 255){
					*counts_COM_is_beijing = 0;
					*counts_COM_is_qianjing += 1;
				}
				else if (object_cunzai.data[nPxIter] == 125){
					*counts_COM_is_qianjing = 0;
					*counts_COM_is_beijing += 1;
				}
				else{
					*counts_COM_is_beijing = 0;
					*counts_COM_is_qianjing = 0;
				}
			}
			for (size_t nModelIter = 0; nModelIter < m_nTotPxCount; ++nModelIter){
				const size_t nPxIter = m_aPxIdxLUT[nModelIter];
				const int nCurrImgCoord_X = m_aPxInfoLUT[nPxIter].nImgCoord_X;
				const int nCurrImgCoord_Y = m_aPxInfoLUT[nPxIter].nImgCoord_Y;
				if ((nCurrImgCoord_X > 1) && (nCurrImgCoord_X < (col - 2)) && (nCurrImgCoord_Y>1) && (nCurrImgCoord_Y < (row - 2))){
					const size_t nPxIterRGB = nPxIter * 3;
					const size_t nDescIterRGB = nPxIterRGB * 2;
					const size_t nDescIter = nPxIter * 2;
					ushort* counts_COM_is_beijing = (ushort*)(COM_is_beijing.data + nDescIter);
					ushort* counts_COM_is_qianjing = (ushort*)(COM_is_qianjing.data + nDescIter);
					ushort* counts_COM_foreground = (ushort*)(COM_foreground.data + nDescIter);
					if (((*counts_COM_is_beijing) >= (ushort)30) && ((*counts_COM_foreground) >= (ushort)30)){
						const uchar* const anCurrColor = CurrImg.data + nPxIterRGB;
						int random = rng.uniform(0, NUMBER);
						ushort anCurrIntraDesc[3];
						const size_t anCurrIntraLBSPThresholds[3] = { m_anLBSPThreshold_8bitLUT[anCurrColor[0]], m_anLBSPThreshold_8bitLUT[anCurrColor[1]], m_anLBSPThreshold_8bitLUT[anCurrColor[2]] };
						LBSP::computeRGBDescriptor(CurrImg, anCurrColor, nCurrImgCoord_X, nCurrImgCoord_Y, anCurrIntraLBSPThresholds, anCurrIntraDesc);
						for (size_t c = 0; c < 3; ++c){
							*(BGColorSamples[random].data + nPxIterRGB + c) = anCurrColor[c];
							*((ushort*)(BGDescSamples[random].data + nDescIterRGB + 2 * c)) = anCurrIntraDesc[c];
						}
					}
				}
			}
			sprintf(fileName7, OutPathPost, k);
			Mat CurrentMaskRecover = Mat::zeros(image.size(), CV_8UC1);
			Recover(CurrFGMask, CurrentMaskRecover, XX, row, col);
			imwrite(fileName7, CurrentMaskRecover);

			addWeighted(CurrMeanFinalSegmRes_LT, (1.0f - fRollAvgFactor_LT), LastFGMask, (1.0 / UCHAR_MAX)*fRollAvgFactor_LT, 0, CurrMeanFinalSegmRes_LT, CV_32F);
			addWeighted(CurrMeanFinalSegmRes_ST, (1.0f - fRollAvgFactor_ST), LastFGMask, (1.0 / UCHAR_MAX)*fRollAvgFactor_ST, 0, CurrMeanFinalSegmRes_ST, CV_32F);
			const float fCurrNonZeroDescRatio = (float)nNonZeroDescCount / (row*col);
			if (fCurrNonZeroDescRatio<0.100f && m_fLastNonZeroDescRatio < 0.100f){
				for (size_t t = 0; t <= UCHAR_MAX; ++t)
					if (m_anLBSPThreshold_8bitLUT[t] > saturate_cast<uchar>(0.000f + ceil(t*refs / 4)))
						--m_anLBSPThreshold_8bitLUT[t];
			}
			else if (fCurrNonZeroDescRatio>0.500f && m_fLastNonZeroDescRatio > 0.500f){
				for (size_t t = 0; t <= UCHAR_MAX; ++t)
					if (m_anLBSPThreshold_8bitLUT[t] < cv::saturate_cast<uchar>(0.000f + UCHAR_MAX*refs))
						++m_anLBSPThreshold_8bitLUT[t];
			}
			m_fLastNonZeroDescRatio = fCurrNonZeroDescRatio;
			if (m_bLearningRateScalingEnabled) {
				resize(CurrImg, m_oDownSampledFrame_MotionAnalysis, m_oDownSampledFrameSize, 0, 0, cv::INTER_AREA);
				accumulateWeighted(m_oDownSampledFrame_MotionAnalysis, m_oMeanDownSampledLastDistFrame_LT, fRollAvgFactor_LT);
				accumulateWeighted(m_oDownSampledFrame_MotionAnalysis, m_oMeanDownSampledLastDistFrame_ST, fRollAvgFactor_ST);
				size_t nTotColorDiff = 0;
				for (int i = 0; i < m_oMeanDownSampledLastDistFrame_ST.rows; ++i) {
					const size_t idx1 = m_oMeanDownSampledLastDistFrame_ST.step.p[0] * i;
					for (int j = 0; j < m_oMeanDownSampledLastDistFrame_ST.cols; ++j) {
						const size_t idx2 = idx1 + m_oMeanDownSampledLastDistFrame_ST.step.p[1] * j;
						nTotColorDiff += (m_nImgChannels == 1) ?
							(size_t)fabs((*(float*)(m_oMeanDownSampledLastDistFrame_ST.data + idx2)) - (*(float*)(m_oMeanDownSampledLastDistFrame_LT.data + idx2))) / 2
							:  //(m_nImgChannels==3)
							std::max((size_t)fabs((*(float*)(m_oMeanDownSampledLastDistFrame_ST.data + idx2)) - (*(float*)(m_oMeanDownSampledLastDistFrame_LT.data + idx2))),
							std::max((size_t)fabs((*(float*)(m_oMeanDownSampledLastDistFrame_ST.data + idx2 + 4)) - (*(float*)(m_oMeanDownSampledLastDistFrame_LT.data + idx2 + 4))),
							(size_t)fabs((*(float*)(m_oMeanDownSampledLastDistFrame_ST.data + idx2 + 8)) - (*(float*)(m_oMeanDownSampledLastDistFrame_LT.data + idx2 + 8)))));
					}
				}
				const float fCurrColorDiffRatio = (float)nTotColorDiff / (m_oMeanDownSampledLastDistFrame_ST.rows*m_oMeanDownSampledLastDistFrame_ST.cols);
				//cout << "ceshi： " << fCurrColorDiffRatio << endl;
				if (m_bAutoModelResetEnabled) {
					if (m_nFramesSinceLastReset > 1000)
						m_bAutoModelResetEnabled = false;
					else if (fCurrColorDiffRatio >= 10.0f  && m_nModelResetCooldown == 0) {
						m_nFramesSinceLastReset = 0;
						const size_t nModelsToRefresh = (size_t)(0.1f*NUMBER);
						const size_t nRefreshStartPos = rand() % NUMBER;
						for (size_t nModelIter = 0; nModelIter < m_nTotPxCount; ++nModelIter){
							const size_t nPxIter = m_aPxIdxLUT[nModelIter];
							const int nCurrImgCoord_X = m_aPxInfoLUT[nPxIter].nImgCoord_X;
							const int nCurrImgCoord_Y = m_aPxInfoLUT[nPxIter].nImgCoord_Y;
							if ((nCurrImgCoord_X > 1) && (nCurrImgCoord_X < (col - 2)) && (nCurrImgCoord_Y>1) && (nCurrImgCoord_Y < (row - 2))){
								if (!LastFGMask.data[nPxIter]) {
									for (size_t nCurrModelIdx = nRefreshStartPos; nCurrModelIdx < nRefreshStartPos + nModelsToRefresh; ++nCurrModelIdx) {
										int nSampleImgCoord_Y, nSampleImgCoord_X;
										getRandSamplePosition(nSampleImgCoord_X, nSampleImgCoord_Y, m_aPxInfoLUT[nPxIter].nImgCoord_X, m_aPxInfoLUT[nPxIter].nImgCoord_Y, LBSP::PATCH_SIZE / 2, ImgSize);
										const size_t nSamplePxIdx = ImgSize.width*nSampleImgCoord_Y + nSampleImgCoord_X;
										if (!LastFGMask.data[nSamplePxIdx]) {
											const size_t nCurrRealModelIdx = nCurrModelIdx%NUMBER;
											for (size_t c = 0; c < 3; ++c) {
												BGColorSamples[nCurrRealModelIdx].data[nPxIter * 3 + c] = LastColorFrame.data[nSamplePxIdx * 3 + c];
												*((ushort*)(BGDescSamples[nCurrRealModelIdx].data + (nPxIter * 3 + c) * 2)) = *((ushort*)(LastDescFrame.data + (nSamplePxIdx * 3 + c) * 2));
											}
										}
									}
								}
							}
							m_nModelResetCooldown = 15;
							//UpdateRateFrame = cv::Scalar(1.0f);
						}
					}
					else
						++m_nFramesSinceLastReset;
				}
				else if (fCurrColorDiffRatio >= 12.0f) {
					m_nFramesSinceLastReset = 0;
					m_bAutoModelResetEnabled = true;
				}
				if (m_nModelResetCooldown > 0)
					--m_nModelResetCooldown;
			}
			cout << "帧数： " << k << endl;
		}

		else{ // 是单通道
			RNG rng;
			sprintf(fileName4, input, k);
			image4 = imread(fileName4, 0);
			//Mat srcImg44 = Mat::zeros(image.size(), CV_8UC1);
			//cvtColor(image4, srcImg44, CV_BGR2GRAY);
			doudong(imageBase, image4, XX);
			//cout << XX << endl;
			Mat CurrImg = Mat::zeros(image.size(), CV_8UC1);
			JiaoZhengGray(image4, CurrImg, XX, row, col);
			size_t nNonZeroDescCount = 0;
			const float fRollAvgFactor_LT = 1.0f / min(++FrameIndex, 100);
			const float fRollAvgFactor_ST = 1.0f / min(FrameIndex, 25);
			for (size_t nModelIter = 0; nModelIter < m_nTotPxCount; ++nModelIter){
				const size_t nPxIter = m_aPxIdxLUT[nModelIter];
				const int nCurrImgCoord_X = m_aPxInfoLUT[nPxIter].nImgCoord_X;
				const int nCurrImgCoord_Y = m_aPxInfoLUT[nPxIter].nImgCoord_Y;
				if ((nCurrImgCoord_X > 1) && (nCurrImgCoord_X < col - 2) && (nCurrImgCoord_Y>1) && (nCurrImgCoord_Y < row - 2)){
					const size_t nDescIter = nPxIter * 2;
					const size_t nFloatIter = nPxIter * 4;
					const uchar anCurrColor = CurrImg.data[nPxIter];
					float *pfCurrVariationFactor = (float*)(CurrVariationFactor.data + nFloatIter);
					float* pfCurrDistThresholdFactor = (float*)(CurrDistThresholdFactor.data + nFloatIter);
					float* pfCurrLearningRate = ((float*)(UpdateRateFrame.data + nFloatIter));
					float* pfCurrMeanLastDist = ((float*)(CurrMeanLastDist.data + nFloatIter));
					float* pfCurrMeanMinDist_LT = ((float*)(CurrMeanMinDist_LT.data + nFloatIter));
					float* pfCurrMeanMinDist_ST = ((float*)(CurrMeanMinDist_ST.data + nFloatIter));
					float* pfCurrMeanRawSegmRes_LT = ((float*)(CurrMeanRawSegmRes_LT.data + nFloatIter));
					float* pfCurrMeanRawSegmRes_ST = ((float*)(CurrMeanRawSegmRes_ST.data + nFloatIter));
					float* pfCurrMeanFinalSegmRes_LT = ((float*)(CurrMeanFinalSegmRes_LT.data + nFloatIter));
					float* pfCurrMeanFinalSegmRes_ST = ((float*)(CurrMeanFinalSegmRes_ST.data + nFloatIter));
					ushort& LastIntraDesc = *((ushort*)(LastDescFrame.data + nDescIter));
					uchar& LastColor = LastColorFrame.data[nPxIter];
					size_t nCurrColorDistThreshold = (size_t)(*pfCurrDistThresholdFactor*30.0f) / 2;
					if (nCurrColorDistThreshold > 35)
						nCurrColorDistThreshold = 35;
					//size_t nCurrDescDistThreshold = (((size_t)1 << ((size_t)floor(*pfCurrDistThresholdFactor + 0.5f)))) * 3;
					size_t nCurrDescDistThreshold = (size_t)round((pow(2, *pfCurrDistThresholdFactor) + 3));
					if (nCurrDescDistThreshold > 7)
						nCurrDescDistThreshold = 7;
					ushort anCurrInterDesc, anCurrIntraDesc;
					LBSP::computeGrayscaleDescriptor(CurrImg, anCurrColor, nCurrImgCoord_X, nCurrImgCoord_Y, m_anLBSPThreshold_8bitLUT[anCurrColor], anCurrIntraDesc);
					size_t nMinTotDescDist = 16;
					size_t nMinTotSumDist = 255;
					UnstableRegionMask.data[nPxIter] = ((*pfCurrDistThresholdFactor) > 2.000f || (*pfCurrMeanRawSegmRes_LT - *pfCurrMeanFinalSegmRes_LT) > 0.100f || (*pfCurrMeanRawSegmRes_ST - *pfCurrMeanFinalSegmRes_ST) > 0.100f) ? 1 : 0;
					size_t GoodSamplesCount = 0, SampleIdx = 0;
					while (GoodSamplesCount < 2 && SampleIdx < NUMBER){
						const uchar& anBGColor = BGColorSamples[SampleIdx].data[nPxIter];
						{
							size_t ColorDist = L1dist(anCurrColor, anBGColor);
							if (ColorDist > nCurrColorDistThreshold)
								goto failedcheck1ch;
							const ushort& anBGIntraDesc = *((ushort*)(BGDescSamples[SampleIdx].data + nDescIter));
							LBSP::computeGrayscaleDescriptor(CurrImg, anBGColor, nCurrImgCoord_X, nCurrImgCoord_Y, m_anLBSPThreshold_8bitLUT[anBGColor], anCurrInterDesc);
							size_t LbspDist_Intra = hdist(anCurrIntraDesc, anBGIntraDesc);
							size_t LbspDist_Inter = hdist(anCurrInterDesc, anBGIntraDesc);
							size_t LbspDist = (size_t)(LbspDist_Inter + LbspDist_Intra) / 2;
							//size_t LbspDist = LbspDist_Inter;
							if (LbspDist > nCurrDescDistThreshold)
								goto failedcheck1ch;
							if (nMinTotDescDist > ColorDist)
								nMinTotDescDist = ColorDist;
							if (nMinTotSumDist > LbspDist)
								nMinTotSumDist = LbspDist;
							GoodSamplesCount++;
						}
					failedcheck1ch:
						SampleIdx++;
					}
					if (GoodSamplesCount < 2){
						const float fNormalizedMinDist = min(1.0f, ((float)nMinTotSumDist / 255 + (float)nMinTotDescDist / 16) / 2 + (float)((2 - GoodSamplesCount) / 2));
						*pfCurrMeanMinDist_LT = (*pfCurrMeanMinDist_LT)*(1.0f - fRollAvgFactor_LT) + fNormalizedMinDist*fRollAvgFactor_LT;
						*pfCurrMeanMinDist_ST = (*pfCurrMeanMinDist_ST)*(1.0f - fRollAvgFactor_ST) + fNormalizedMinDist*fRollAvgFactor_ST;
						*pfCurrMeanRawSegmRes_LT = (*pfCurrMeanRawSegmRes_LT)*(1.0f - fRollAvgFactor_LT) + fRollAvgFactor_LT;
						*pfCurrMeanRawSegmRes_ST = (*pfCurrMeanRawSegmRes_ST)*(1.0f - fRollAvgFactor_ST) + fRollAvgFactor_ST;
						CurrFGMask.data[nPxIter] = UCHAR_MAX;
					}
					else{
						const float fNormalizedMinDist = ((float)(nMinTotSumDist / 255) + (float)(nMinTotDescDist / 16)) / 2;
						*pfCurrMeanMinDist_LT = (*pfCurrMeanMinDist_LT)*(1.0f - fRollAvgFactor_LT) + fNormalizedMinDist*fRollAvgFactor_LT;
						*pfCurrMeanMinDist_ST = (*pfCurrMeanMinDist_ST)*(1.0f - fRollAvgFactor_ST) + fNormalizedMinDist*fRollAvgFactor_ST;
						*pfCurrMeanRawSegmRes_LT = (*pfCurrMeanRawSegmRes_LT)*(1.0f - fRollAvgFactor_LT);
						*pfCurrMeanRawSegmRes_ST = (*pfCurrMeanRawSegmRes_ST)*(1.0f - fRollAvgFactor_ST);
						CurrFGMask.data[nPxIter] = 0;
						size_t Tx = (size_t)round(*pfCurrLearningRate);
						size_t random = rng.uniform(0, Tx);
						if (random == 1){
							random = rng.uniform(0, NUMBER);
							*((ushort*)(BGDescSamples[random].data + nDescIter)) = anCurrIntraDesc;
							BGColorSamples[random].data[nPxIter] = anCurrColor;
						}
					}
					if (LastFGMask.data[nPxIter] || (min(*pfCurrMeanMinDist_LT, *pfCurrMeanMinDist_ST) < 0.100f && CurrFGMask.data[nPxIter])){
						if (*pfCurrLearningRate<20.0f)
							*pfCurrLearningRate += 0.5000f / (max(*pfCurrMeanMinDist_LT, *pfCurrMeanMinDist_ST)*(*pfCurrVariationFactor));
					}
					else if (*pfCurrLearningRate>2.0000f)
						*pfCurrLearningRate -= 0.2500f*(*pfCurrVariationFactor) / max(*pfCurrMeanMinDist_LT, *pfCurrMeanMinDist_ST);
					if (*pfCurrLearningRate<2.0000f)
						*pfCurrLearningRate = 2.0000f;
					else if (*pfCurrLearningRate>20.00f)
						*pfCurrLearningRate = 20.00f;
					//if (max(*pfCurrMeanMinDist_LT, *pfCurrMeanMinDist_ST)>0.100f && BlinksFrame.data[nPxIter])
					if (BlinksFrame.data[nPxIter])
						(*pfCurrVariationFactor) += 1.0000f;
					else if ((*pfCurrVariationFactor) > 0.100f){
						*pfCurrVariationFactor -= LastFGMask.data[nPxIter] ? 0.025f : UnstableRegionMask.data[nPxIter] ? 0.050f : 0.100f;
						if ((*pfCurrVariationFactor) < 0.100f)
							(*pfCurrVariationFactor) = 0.100f;
					}
					if ((*pfCurrDistThresholdFactor) < pow(1.0f + min(*pfCurrMeanMinDist_LT, *pfCurrMeanMinDist_ST) * 2, 2)){
						(*pfCurrDistThresholdFactor) += 0.01f*((*pfCurrVariationFactor) - 0.100f);
					}
					else{
						(*pfCurrDistThresholdFactor) -= 0.01f / (*pfCurrVariationFactor);
						if ((*pfCurrDistThresholdFactor) < 1.0f)
							(*pfCurrDistThresholdFactor) = 1.0f;
					}
					if (popcount(anCurrIntraDesc) >= 2)
						++nNonZeroDescCount;
					LastIntraDesc = anCurrIntraDesc;
					LastColor = anCurrColor;
				}
			}
			//sprintf(fileName5, OutPath, k);
			//imwrite(fileName5, CurrFGMask);
			medianBlur(CurrFGMask, CurrFGMask, 3);
			medianBlur(LastRawFGMask, LastRawFGMask, 3);
			bitwise_xor(CurrFGMask, LastRawFGMask, CurrRawFGBlinkMask);
			bitwise_or(CurrRawFGBlinkMask, LastRawFGBlinkMask, BlinksFrame);
			CurrRawFGBlinkMask.copyTo(LastRawFGBlinkMask);
			CurrFGMask.copyTo(LastRawFGMask);
			morphologyEx(CurrFGMask, CurrFGMask_close, MORPH_CLOSE, Mat());
			CurrFGMask_close.copyTo(CurrFGMask_close_fill);
			floodFill(CurrFGMask_close_fill, Point(0, 0), UCHAR_MAX);
			bitwise_not(CurrFGMask_close_fill, CurrFGMask_close_fill);
			erode(CurrFGMask_close, CurrFGMask_close, Mat(), Point(-1, -1), 3);
			bitwise_or(CurrFGMask, CurrFGMask_close_fill, CurrFGMask);
			bitwise_or(CurrFGMask, CurrFGMask_close, CurrFGMask);
			medianBlur(CurrFGMask, LastFGMask, m_nMedianBlurKernelSize);
			dilate(LastFGMask, LastFGMask_dilated, Mat(), Point(-1, -1), 3);
			bitwise_and(BlinksFrame, LastFGMask_dilated_inverted, BlinksFrame);
			bitwise_not(LastFGMask_dilated, LastFGMask_dilated_inverted);
			bitwise_and(BlinksFrame, LastFGMask_dilated_inverted, BlinksFrame);
			LastFGMask.copyTo(CurrFGMask);
			Mat object_cunzai = Mat::zeros(image.size(), CV_8UC1);
			cunzai(RefBgEdge, CurrImg, CurrFGMask, image, object_cunzai, row, col);
			for (size_t nModelIter = 0; nModelIter < m_nTotPxCount; ++nModelIter){
				const size_t nPxIter = m_aPxIdxLUT[nModelIter];
				const size_t nDescIter = nPxIter * 2;
				ushort* counts_COM_foreground = (ushort*)(COM_foreground.data + nDescIter);
				if (CurrFGMask.data[nPxIter] == 255)
					* counts_COM_foreground += 1;
				else
					* counts_COM_foreground = 0;
			}
			for (size_t nModelIter = 0; nModelIter < m_nTotPxCount; ++nModelIter){
				const size_t nPxIter = m_aPxIdxLUT[nModelIter];
				const size_t nDescIter = nPxIter * 2;
				ushort* counts_COM_is_beijing = (ushort*)(COM_is_beijing.data + nDescIter);
				ushort* counts_COM_is_qianjing = (ushort*)(COM_is_qianjing.data + nDescIter);
				if (object_cunzai.data[nPxIter] == 255){
					*counts_COM_is_beijing = 0;
					*counts_COM_is_qianjing += 1;
				}
				else if (object_cunzai.data[nPxIter] == 125){
					*counts_COM_is_qianjing = 0;
					*counts_COM_is_beijing += 1;
				}
				else{
					*counts_COM_is_beijing = 0;
					*counts_COM_is_qianjing = 0;
				}
			}
			for (size_t nModelIter = 0; nModelIter < m_nTotPxCount; ++nModelIter){
				const size_t nPxIter = m_aPxIdxLUT[nModelIter];
				const int nCurrImgCoord_X = m_aPxInfoLUT[nPxIter].nImgCoord_X;
				const int nCurrImgCoord_Y = m_aPxInfoLUT[nPxIter].nImgCoord_Y;
				if ((nCurrImgCoord_X > 1) && (nCurrImgCoord_X < (col - 2)) && (nCurrImgCoord_Y>1) && (nCurrImgCoord_Y < (row - 2))){
					const size_t nDescIter = nPxIter * 2;
					ushort* counts_COM_is_beijing = (ushort*)(COM_is_beijing.data + nDescIter);
					ushort* counts_COM_is_qianjing = (ushort*)(COM_is_qianjing.data + nDescIter);
					ushort* counts_COM_foreground = (ushort*)(COM_foreground.data + nDescIter);
					if (((*counts_COM_is_beijing) >= (ushort)30) && ((*counts_COM_foreground) >= (ushort)30)){
						const uchar anCurrColor = CurrImg.data[nPxIter];
						ushort anCurrIntraDesc;
						LBSP::computeGrayscaleDescriptor(CurrImg, anCurrColor, nCurrImgCoord_X, nCurrImgCoord_Y, m_anLBSPThreshold_8bitLUT[anCurrColor], anCurrIntraDesc);
						int random = rng.uniform(0, NUMBER);
						*((ushort*)(BGDescSamples[random].data + nDescIter)) = anCurrIntraDesc;
						BGColorSamples[random].data[nPxIter] = anCurrColor;
					}
				}
			}
			sprintf(fileName7, OutPathPost, k);
			Mat CurrentMaskRecover = Mat::zeros(image.size(), CV_8UC1);
			Recover(CurrFGMask, CurrentMaskRecover, XX, row, col);
			imwrite(fileName7, CurrentMaskRecover);

			addWeighted(CurrMeanFinalSegmRes_LT, (1.0f - fRollAvgFactor_LT), LastFGMask, (1.0 / UCHAR_MAX)*fRollAvgFactor_LT, 0, CurrMeanFinalSegmRes_LT, CV_32F);
			addWeighted(CurrMeanFinalSegmRes_ST, (1.0f - fRollAvgFactor_ST), LastFGMask, (1.0 / UCHAR_MAX)*fRollAvgFactor_ST, 0, CurrMeanFinalSegmRes_ST, CV_32F);
			const float fCurrNonZeroDescRatio = (float)nNonZeroDescCount / (row*col);
			if (fCurrNonZeroDescRatio<0.100f && m_fLastNonZeroDescRatio < 0.100f){
				for (size_t t = 0; t <= UCHAR_MAX; ++t)
					if (m_anLBSPThreshold_8bitLUT[t] > saturate_cast<uchar>(0.000f + ceil(t*refs / 4)))
						--m_anLBSPThreshold_8bitLUT[t];
			}
			else if (fCurrNonZeroDescRatio>0.500f && m_fLastNonZeroDescRatio > 0.500f){
				for (size_t t = 0; t <= UCHAR_MAX; ++t)
					if (m_anLBSPThreshold_8bitLUT[t] < cv::saturate_cast<uchar>(0.000f + UCHAR_MAX*refs))
						++m_anLBSPThreshold_8bitLUT[t];
			}
			m_fLastNonZeroDescRatio = fCurrNonZeroDescRatio;

			if (m_bLearningRateScalingEnabled) {
				resize(CurrImg, m_oDownSampledFrame_MotionAnalysis, m_oDownSampledFrameSize, 0, 0, cv::INTER_AREA);
				accumulateWeighted(m_oDownSampledFrame_MotionAnalysis, m_oMeanDownSampledLastDistFrame_LT, fRollAvgFactor_LT);
				accumulateWeighted(m_oDownSampledFrame_MotionAnalysis, m_oMeanDownSampledLastDistFrame_ST, fRollAvgFactor_ST);
				size_t nTotColorDiff = 0;
				for (int i = 0; i < m_oMeanDownSampledLastDistFrame_ST.rows; ++i) {
					const size_t idx1 = m_oMeanDownSampledLastDistFrame_ST.step.p[0] * i;
					for (int j = 0; j < m_oMeanDownSampledLastDistFrame_ST.cols; ++j) {
						const size_t idx2 = idx1 + m_oMeanDownSampledLastDistFrame_ST.step.p[1] * j;
						nTotColorDiff += (m_nImgChannels == 1) ?
							(size_t)fabs((*(float*)(m_oMeanDownSampledLastDistFrame_ST.data + idx2)) - (*(float*)(m_oMeanDownSampledLastDistFrame_LT.data + idx2))) / 2
							:  //(m_nImgChannels==3)
							std::max((size_t)fabs((*(float*)(m_oMeanDownSampledLastDistFrame_ST.data + idx2)) - (*(float*)(m_oMeanDownSampledLastDistFrame_LT.data + idx2))),
							std::max((size_t)fabs((*(float*)(m_oMeanDownSampledLastDistFrame_ST.data + idx2 + 4)) - (*(float*)(m_oMeanDownSampledLastDistFrame_LT.data + idx2 + 4))),
							(size_t)fabs((*(float*)(m_oMeanDownSampledLastDistFrame_ST.data + idx2 + 8)) - (*(float*)(m_oMeanDownSampledLastDistFrame_LT.data + idx2 + 8)))));
					}
				}
				const float fCurrColorDiffRatio = (float)nTotColorDiff / (m_oMeanDownSampledLastDistFrame_ST.rows*m_oMeanDownSampledLastDistFrame_ST.cols);
				//cout << "ceshi： " << fCurrColorDiffRatio << endl;
				if (m_bAutoModelResetEnabled) {
					if (m_nFramesSinceLastReset > 1000)
						m_bAutoModelResetEnabled = false;
					else if (fCurrColorDiffRatio >= 10.0f  && m_nModelResetCooldown == 0) {
						m_nFramesSinceLastReset = 0;
						const size_t nModelsToRefresh = (size_t)(0.1f*NUMBER);
						const size_t nRefreshStartPos = rand() % NUMBER;
						for (size_t nModelIter = 0; nModelIter < m_nTotPxCount; ++nModelIter){
							const size_t nPxIter = m_aPxIdxLUT[nModelIter];
							const int nCurrImgCoord_X = m_aPxInfoLUT[nPxIter].nImgCoord_X;
							const int nCurrImgCoord_Y = m_aPxInfoLUT[nPxIter].nImgCoord_Y;
							if ((nCurrImgCoord_X > 1) && (nCurrImgCoord_X < (col - 2)) && (nCurrImgCoord_Y>1) && (nCurrImgCoord_Y < (row - 2))){
								if (!LastFGMask.data[nPxIter]) {
									for (size_t nCurrModelIdx = nRefreshStartPos; nCurrModelIdx < nRefreshStartPos + nModelsToRefresh; ++nCurrModelIdx) {
										int nSampleImgCoord_Y, nSampleImgCoord_X;
										getRandSamplePosition(nSampleImgCoord_X, nSampleImgCoord_Y, m_aPxInfoLUT[nPxIter].nImgCoord_X, m_aPxInfoLUT[nPxIter].nImgCoord_Y, LBSP::PATCH_SIZE / 2, ImgSize);
										const size_t nSamplePxIdx = ImgSize.width*nSampleImgCoord_Y + nSampleImgCoord_X;
										if (!LastFGMask.data[nSamplePxIdx]) {
											const size_t nCurrRealModelIdx = nCurrModelIdx%NUMBER;
											BGColorSamples[nCurrRealModelIdx].data[nPxIter] = LastColorFrame.data[nSamplePxIdx];
											*((ushort*)(BGDescSamples[nCurrRealModelIdx].data + nPxIter * 2)) = *((ushort*)(LastDescFrame.data + nSamplePxIdx * 2));
										}
									}
								}
							}
							m_nModelResetCooldown = 15;
							//UpdateRateFrame = cv::Scalar(1.0f);
						}
					}
					else
						++m_nFramesSinceLastReset;
				}
				else if (fCurrColorDiffRatio >= 12.0f) {
					m_nFramesSinceLastReset = 0;
					m_bAutoModelResetEnabled = true;
				}
				if (m_nModelResetCooldown > 0)
					--m_nModelResetCooldown;
			}
			cout << "TotFrames： " << k << endl;
		}
	}
}


int main(){

	double time0;
	time0 = (double)getTickCount();

	const char* input_highway = "D:\\CDnet2012\\BaseLine\\highway\\input\\%06d.jpg";
	const char* ROI_highway = "D:\\CDnet2012\\BaseLine\\highway\\%06d.bmp";
	const int TotFrames_highway = 1700;
	const int P_highway = 4;
	const float ref_highway = 0.27f;
	const char* OutPathPost_highway = "D:\\CDnet2012\\BaseLine\\result\\%06d.png";
	MyBS(input_highway, OutPathPost_highway, TotFrames_highway, P_highway, ref_highway, ROI_highway);

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	time0 = ((double)getTickCount() - time0) / getTickFrequency();
	cout << "运行时间：" << time0 << "秒" << endl;
	waitKey(0);
	return 0;
}
