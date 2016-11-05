/*
 * CannyEdgeDetector.h
 *
 *  Created on: Mar 6, 2014
 *      Author: Yasin Yıldırım, (at Simit Lab)
 */

#ifndef CANNYEDGEDETECTOR_H_
#define CANNYEDGEDETECTOR_H_

#include <iostream>
#include <cstdio>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <cmath>

//#include <cstdlib>
using namespace std;
using namespace cv;


struct Image{
	unsigned char* imgPtr;
	int height;
	int width;
};
typedef struct Image Image;

class CannyEdgeDetector {
public:
	Mat imgGray;
	Mat imgGray_copy;
	Mat Dx;
	Mat Dy;
	Mat D;
	Mat D_new;
	Mat Teta;
	Mat thresh;
	Mat dest;

	CannyEdgeDetector();
	CannyEdgeDetector(Mat originalImage);
	virtual ~CannyEdgeDetector();

	void detect(int lowThreshold, int highThreshold );
	void detect();
};

#endif /* CANNYEDGEDETECTOR_H_ */
