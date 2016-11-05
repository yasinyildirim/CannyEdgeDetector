/*
 * main.cpp
 *
 *  Created on: Mar 7, 2014
 *      Author: Yasin Yıldırım @ Simit Lab
 */


#include "CannyEdgeDetector.h"
Mat src;
const char* source_window = " Canny result Image";

int main(int, char** argv) {
	/// Load source image and convert it to gray

	  src = imread( argv[1], 1 );


	  /// Create Window
	  namedWindow( source_window, CV_WINDOW_AUTOSIZE );

	  /// Create Trackbar to set the number of corners
	  CannyEdgeDetector ced(src);
	  ced.detect();



	  //result image of step 3
	  imshow( "Threshold without Hysteresis", ced.thresh );
	  //grayscale of source image
	  imshow( " Grayscale", ced.imgGray_copy );

	  //canny edge detected image
	  imshow( source_window, ced.dest );

	  waitKey(0);
	return 0;
}

