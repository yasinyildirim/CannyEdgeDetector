/*
 * CannyEdgeDetector.cpp
 *
 *  Created on: Mar 6, 2014
 *      Author: Yasin Yıldırım @ Simit Lab
 */

#include "CannyEdgeDetector.h"

CannyEdgeDetector::CannyEdgeDetector() {
	// TODO Auto-generated constructor stub

}

CannyEdgeDetector::CannyEdgeDetector(Mat src) {
	// Load original image and store it as a grayscale image

	imgGray = Mat::zeros(src.rows, src.cols, CV_8UC1);
	dest = Mat::zeros(src.rows, src.cols, CV_8UC1);
	Dx = Mat::zeros(src.rows, src.cols, CV_16SC1);
	Dy = Mat::zeros(src.rows, src.cols, CV_16SC1);
	D = Mat::zeros(src.rows, src.cols, CV_32FC1);
	Teta = Mat::zeros(src.rows, src.cols, CV_32FC1);
	thresh = Mat::zeros(src.rows, src.cols, CV_8UC1);
	D_new = Mat::zeros(src.rows, src.cols, CV_32FC1);
	imgGray_copy;

	const int channels = src.channels();

	// on the fly matrix element access
	switch (channels) {
	case 1: {
		for (int i = 0; i < src.rows; ++i) {
			for (int j = 0; j < src.cols; ++j) {
				imgGray.at<uchar>(i, j) = src.at<uchar>(i, j);
			}
		}

		break;
	}
	case 3: {
		Mat_<Vec3b> _I = src;

		for (int i = 0; i < src.rows; ++i)
			for (int j = 0; j < src.cols; ++j) {
				imgGray.at<uchar>(i, j) = (uchar) (_I(i, j)[0] * 0.114 + _I(i, j)[1]
						* 0.587 + _I(i, j)[2] * 0.299);
			}
		src = _I;
	}
	}
	imgGray_copy = imgGray.clone();

}

CannyEdgeDetector::~CannyEdgeDetector() {
	// TODO Auto-generated destructor stub

}

void CannyEdgeDetector::detect() {

	const int BLOCK_SIZE = 5;
	const int BLOCK_SIZE3X = 3;
	int rowBlocks = imgGray.rows - BLOCK_SIZE;
	int colBlocks = imgGray.cols - BLOCK_SIZE;

	int rowBlocks3x = imgGray.rows - BLOCK_SIZE3X;
	int colBlocks3x = imgGray.cols - BLOCK_SIZE3X;
	// noise reduction
	float GaussianKernel[5][5] = {
			{ 2 , 4 , 5 , 4 , 2  }, { 4 , 9 ,
					12 , 9 , 4  }, { 5 , 12 , 15 ,
					12 , 5  }, { 4 , 9 , 12 , 9 ,
					4  }, { 2 , 4 , 5 , 4 , 2 } };
	int DxKernel[3][3] = { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } };
	int DyKernel[3][3] = { { 1, 2, 1 }, { 0, 0, 0 }, { -1, -2, -1 } };

	//Step 1: Noise Reduction
	for (int i = 0; i < rowBlocks; i++) {
		for (int j = 0; j < colBlocks; j++) {
			Mat windowImage_ij(imgGray, Rect(j, i, BLOCK_SIZE, BLOCK_SIZE));
			// block processing loops
			float gbValues[5][5] = { { 0 } }; // Gaussian blur resulting float values
			vector<float> vgbValues;
			for (int k = 0; k < windowImage_ij.rows; k++) {

				for (int m = 0; m < windowImage_ij.cols; m++) {
					gbValues[k][m] = windowImage_ij.at<uchar>(k, m)
							* GaussianKernel[k][m];
					vgbValues.push_back((windowImage_ij.at<uchar>(k, m)
							* GaussianKernel[k][m]));
				}
			}
			for (int k = 0; k < windowImage_ij.rows; k++) {

				for (int m = 0; m < windowImage_ij.cols; m++) {
					if (k == 2 && m == 2)
						continue;
					gbValues[2][2] += gbValues[k][m];
					vgbValues[12] += gbValues[k][m];
				}
			}

			imgGray.at<uchar>(i + 2, j + 2) = (uchar) (vgbValues[12] / 159);
			vgbValues.clear();
		}
	}

	//Step 2: Compute Gradient Magnitude and Angle
	for (int i = 0; i < rowBlocks3x; i++) {
		for (int j = 0; j < colBlocks3x; j++) {
			Mat windowImagex_ij(imgGray, Rect(j, i, BLOCK_SIZE3X, BLOCK_SIZE3X));
			Mat windowImagey_ij(imgGray, Rect(j, i, BLOCK_SIZE3X, BLOCK_SIZE3X));
			// block processing loops
			for (int k = 0; k < windowImagex_ij.rows; k++) {

				for (int m = 0; m < windowImagex_ij.cols; m++) {
					windowImagex_ij.at<int>(k, m) = windowImagex_ij.at<int>(k,
							m) * DxKernel[k][m];
					windowImagey_ij.at<int>(k, m) = windowImagey_ij.at<int>(k,
							m) * DyKernel[k][m];
				}
			}

			for (int k = 0; k < windowImagex_ij.rows; k++) {

				for (int m = 0; m < windowImagex_ij.cols; m++) {
					if (k == 1 && m == 1)
						continue;
					windowImagex_ij.at<int>(1, 1) += windowImagex_ij.at<int>(k,
							m);
					windowImagey_ij.at<int>(1, 1) += windowImagey_ij.at<int>(k,
							m);
				}
			}

			Dx.at<int>(i + 1, j + 1) = windowImagex_ij.at<int>(1, 1);
			Dy.at<int>(i + 1, j + 1) = windowImagey_ij.at<int>(1, 1);
			D.at<float>(i + 1, j + 1) = hypot(Dx.at<int>(i + 1, j + 1),
					Dy.at<int>(i + 1, j + 1));
			if (Dy.at<int>(i + 1, j + 1) == 0) {
				Teta.at<float>(i + 1, j + 1) = 90;
			} else {
				Teta.at<float>(i + 1, j + 1) = atan(
						(Dy.at<int>(i + 1, j + 1) / Dx.at<int>(i + 1, j + 1)));
			}
			if (Teta.at<float>(i + 1, j + 1) > 180) {
				Teta.at<float>(i + 1, j + 1) -= 180;
			} else if (Teta.at<float>(i + 1, j + 1) < 0) {
				Teta.at<float>(i + 1, j + 1) += 180;
			}

			if (Teta.at<float>(i + 1, j + 1) <= 22.5
					|| Teta.at<float>(i + 1, j + 1) >= 157.5) {
				Teta.at<float>(i + 1, j + 1) = 0;
			} else if (Teta.at<float>(i + 1, j + 1) > 22.5
					&& Teta.at<float>(i + 1, j + 1) <= 67.5) {
				Teta.at<float>(i + 1, j + 1) = 45;
			} else if (Teta.at<float>(i + 1, j + 1) > 67.5
					&& Teta.at<float>(i + 1, j + 1) <= 112.5) {
				Teta.at<float>(i + 1, j + 1) = 90;
			} else if (Teta.at<float>(i + 1, j + 1) > 112.5
					&& Teta.at<float>(i + 1, j + 1) < 157.5) {
				Teta.at<float>(i + 1, j + 1) = 135;
			}
		}
	}

	//Step 3: Non-Maximum Surpression

	for (int i = 0; i < rowBlocks3x; i++) {
		for (int j = 0; j < colBlocks3x; j++) {
			Mat windowImage_ij(D, Rect(j, i, BLOCK_SIZE3X, BLOCK_SIZE3X));
			Mat windowImageTeta_ij(Teta,
					Rect(j, i, BLOCK_SIZE3X, BLOCK_SIZE3X));
			// block processing loops

			if (windowImageTeta_ij.at<float>(1, 1) == 0) {
				if (windowImage_ij.at<float>(1, 1)
						> windowImage_ij.at<float>(2, 1)
						&& windowImage_ij.at<float>(1, 1)
								> windowImage_ij.at<float>(0, 1)) {
					thresh.at<uchar>(i + 1, j + 1) = 255;
					D_new.at<float>(i + 1, j + 1) = windowImage_ij.at<float>(1,
							1);
				}

			} else if (windowImageTeta_ij.at<float>(1, 1) == 45) {
				if (windowImage_ij.at<float>(1, 1)
						> windowImage_ij.at<float>(2, 2)
						&& windowImage_ij.at<float>(1, 1)
								> windowImage_ij.at<float>(0, 0)) {
					thresh.at<uchar>(i + 1, j + 1) = 255;
					D_new.at<float>(i + 1, j + 1) = windowImage_ij.at<float>(1,
							1);
				}
			} else if (windowImageTeta_ij.at<float>(1, 1) == 90) {
				if (windowImage_ij.at<float>(1, 1)
						> windowImage_ij.at<float>(1, 2)
						&& windowImage_ij.at<float>(1, 1)
								> windowImage_ij.at<float>(1, 0)) {
					thresh.at<uchar>(i + 1, j + 1) = 255;
					D_new.at<float>(i + 1, j + 1) = windowImage_ij.at<float>(1,
							1);
				}
			} else if (windowImageTeta_ij.at<float>(1, 1) == 135) {
				if (windowImage_ij.at<float>(1, 1)
						> windowImage_ij.at<float>(2, 0)
						&& windowImage_ij.at<float>(1, 1)
								> windowImage_ij.at<float>(0, 2)) {
					thresh.at<uchar>(i + 1, j + 1) = 255;
					D_new.at<float>(i + 1, j + 1) = windowImage_ij.at<float>(1,
							1);
				}
			}

		}
	}

	//Step 4: Hysteresis Thresholding
	Scalar meanValue = mean(D, thresh);
	float thigh = meanValue[0];
	float tlow = thigh / 2;

	for (int i = 0; i < rowBlocks3x; i++) {
		for (int j = 0; j < colBlocks3x; j++) {
			Mat windowImage_ij(D_new, Rect(j, i, BLOCK_SIZE3X, BLOCK_SIZE3X));
			Mat windowImage5x_ij;
			if (rowBlocks3x + BLOCK_SIZE < imgGray.rows
					&& colBlocks3x + BLOCK_SIZE < imgGray.cols) {
				Mat windowImage5x_ij(D_new, Rect(j, i, BLOCK_SIZE, BLOCK_SIZE));
			}

			// block processing loops
			bool hasHighValue = false;
			bool hasAtLeastOneAverageValue = false;
			if (windowImage_ij.at<float>(1, 1) > thigh) {
				dest.at<uchar>(i + 1, j + 1) = 255;

			} else if (windowImage_ij.at<float>(1, 1) < tlow) {
				continue;
			} else {
				for (int k = 0; k < windowImage_ij.rows; k++) {
					for (int m = 0; m < windowImage_ij.cols; m++) {
						if (k == 1 && m == 1) {
							continue;
						}
						if (windowImage_ij.at<float>(k, m) > thigh) {
							hasHighValue = true;
							break;
						} else if (windowImage_ij.at<float>(k, m) > tlow) {
							hasAtLeastOneAverageValue = true;
						}
					}
				}
				if (hasHighValue) {
					dest.at<uchar>(i + 1, j + 1) = 255;
				} else if (hasAtLeastOneAverageValue
						&& rowBlocks3x + BLOCK_SIZE < imgGray.rows
						&& colBlocks3x + BLOCK_SIZE < imgGray.cols) {
					for (int k = 0; k < windowImage5x_ij.rows; k++) {
						for (int m = 0; m < windowImage5x_ij.cols; m++) {
							if (k == 2 && m == 2) {
								continue;
							}

							if (windowImage5x_ij.at<float>(k, m) > thigh) {
								hasHighValue = true;
								break;
							}

						}
					}

					if (hasHighValue) {
						dest.at<uchar>(i + 1, j + 1) = 255;
					}

				}
			}

		}
	}

}


void CannyEdgeDetector::detect(int lowThreshold, int highThreshold) {

	const int BLOCK_SIZE = 5;
	const int BLOCK_SIZE3X = 3;
	int rowBlocks = imgGray.rows - BLOCK_SIZE;
	int colBlocks = imgGray.cols - BLOCK_SIZE;

	int rowBlocks3x = imgGray.rows - BLOCK_SIZE3X;
	int colBlocks3x = imgGray.cols - BLOCK_SIZE3X;
	// noise reduction
	float GaussianKernel[5][5] = {
			{ 2 / 159, 4 / 159, 5 / 159, 4 / 159, 2 / 159 }, { 4 / 159, 9 / 159,
					12 / 159, 9 / 159, 4 / 159 }, { 5 / 159, 12 / 159, 15 / 159,
					12 / 159, 5 / 159 }, { 4 / 159, 9 / 159, 12 / 159, 9 / 159,
					4 / 159 }, { 2 / 159, 4 / 159, 5 / 159, 4 / 159, 2 / 159 } };
	int DxKernel[3][3] = { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } };
	int DyKernel[3][3] = { { 1, 2, 1 }, { 0, 0, 0 }, { -1, -2, -1 } };

	//Step 1: Noise Reduction
	for (int i = 0; i < rowBlocks; i++) {
		for (int j = 0; j < colBlocks; j++) {
			Mat windowImage_ij(imgGray, Rect(j, i, BLOCK_SIZE, BLOCK_SIZE));
			// block processing loops
			float gbValues[5][5] = { { 0 } }; // Gaussian blur resulting float values
			for (int k = 0; k < windowImage_ij.rows; k++) {

				for (int m = 0; m < windowImage_ij.cols; m++) {
					gbValues[k][m] = windowImage_ij.at<uchar>(k, m)
							* GaussianKernel[k][m];
				}
			}
			for (int k = 0; k < windowImage_ij.rows; k++) {

				for (int m = 0; m < windowImage_ij.cols; m++) {
					if (k == 2 && m == 2)
						continue;
					gbValues[2][2] += gbValues[k][m];
				}
			}

			//imgGray.at<uchar>(i + 2, j + 2) = (uchar) gbValues[2][2];

		}
	}

	//Step 2: Compute Gradient Magnitude and Angle
	for (int i = 0; i < rowBlocks3x; i++) {
		for (int j = 0; j < colBlocks3x; j++) {
			Mat windowImagex_ij(imgGray, Rect(j, i, BLOCK_SIZE3X, BLOCK_SIZE3X));
			Mat windowImagey_ij(imgGray, Rect(j, i, BLOCK_SIZE3X, BLOCK_SIZE3X));
			// block processing loops
			for (int k = 0; k < windowImagex_ij.rows; k++) {

				for (int m = 0; m < windowImagex_ij.cols; m++) {
					windowImagex_ij.at<int>(k, m) = windowImagex_ij.at<int>(k,
							m) * DxKernel[k][m];
					windowImagey_ij.at<int>(k, m) = windowImagey_ij.at<int>(k,
							m) * DyKernel[k][m];
				}
			}

			for (int k = 0; k < windowImagex_ij.rows; k++) {

				for (int m = 0; m < windowImagex_ij.cols; m++) {
					if (k == 1 && m == 1)
						continue;
					windowImagex_ij.at<int>(1, 1) += windowImagex_ij.at<int>(k,
							m);
					windowImagey_ij.at<int>(1, 1) += windowImagey_ij.at<int>(k,
							m);
				}
			}

			Dx.at<int>(i + 1, j + 1) = windowImagex_ij.at<int>(1, 1);
			Dy.at<int>(i + 1, j + 1) = windowImagey_ij.at<int>(1, 1);
			D.at<float>(i + 1, j + 1) = hypot(Dx.at<int>(i + 1, j + 1),
					Dy.at<int>(i + 1, j + 1));
			if (Dy.at<int>(i + 1, j + 1) == 0) {
				Teta.at<float>(i + 1, j + 1) = 90;
			} else {
				Teta.at<float>(i + 1, j + 1) = atan(
						(Dy.at<int>(i + 1, j + 1) / Dx.at<int>(i + 1, j + 1)));
			}
			if (Teta.at<float>(i + 1, j + 1) > 180) {
				Teta.at<float>(i + 1, j + 1) -= 180;
			} else if (Teta.at<float>(i + 1, j + 1) < 0) {
				Teta.at<float>(i + 1, j + 1) += 180;
			}

			if (Teta.at<float>(i + 1, j + 1) <= 22.5
					|| Teta.at<float>(i + 1, j + 1) >= 157.5) {
				Teta.at<float>(i + 1, j + 1) = 0;
			} else if (Teta.at<float>(i + 1, j + 1) > 22.5
					&& Teta.at<float>(i + 1, j + 1) <= 67.5) {
				Teta.at<float>(i + 1, j + 1) = 45;
			} else if (Teta.at<float>(i + 1, j + 1) > 67.5
					&& Teta.at<float>(i + 1, j + 1) <= 112.5) {
				Teta.at<float>(i + 1, j + 1) = 90;
			} else if (Teta.at<float>(i + 1, j + 1) > 112.5
					&& Teta.at<float>(i + 1, j + 1) < 157.5) {
				Teta.at<float>(i + 1, j + 1) = 135;
			}
		}
	}

	//Step 3: Non-Maximum Surpression

	for (int i = 0; i < rowBlocks3x; i++) {
		for (int j = 0; j < colBlocks3x; j++) {
			Mat windowImage_ij(D, Rect(j, i, BLOCK_SIZE3X, BLOCK_SIZE3X));
			Mat windowImageTeta_ij(Teta,
					Rect(j, i, BLOCK_SIZE3X, BLOCK_SIZE3X));
			// block processing loops

			if (windowImageTeta_ij.at<float>(1, 1) == 0) {
				if (windowImage_ij.at<float>(1, 1)
						> windowImage_ij.at<float>(2, 1)
						&& windowImage_ij.at<float>(1, 1)
								> windowImage_ij.at<float>(0, 1)) {
					thresh.at<uchar>(i + 1, j + 1) = 255;
					D_new.at<float>(i + 1, j + 1) = windowImage_ij.at<float>(1,
							1);
				}

			} else if (windowImageTeta_ij.at<float>(1, 1) == 45) {
				if (windowImage_ij.at<float>(1, 1)
						> windowImage_ij.at<float>(2, 2)
						&& windowImage_ij.at<float>(1, 1)
								> windowImage_ij.at<float>(0, 0)) {
					thresh.at<uchar>(i + 1, j + 1) = 255;
					D_new.at<float>(i + 1, j + 1) = windowImage_ij.at<float>(1,
							1);
				}
			} else if (windowImageTeta_ij.at<float>(1, 1) == 90) {
				if (windowImage_ij.at<float>(1, 1)
						> windowImage_ij.at<float>(1, 2)
						&& windowImage_ij.at<float>(1, 1)
								> windowImage_ij.at<float>(1, 0)) {
					thresh.at<uchar>(i + 1, j + 1) = 255;
					D_new.at<float>(i + 1, j + 1) = windowImage_ij.at<float>(1,
							1);
				}
			} else if (windowImageTeta_ij.at<float>(1, 1) == 135) {
				if (windowImage_ij.at<float>(1, 1)
						> windowImage_ij.at<float>(2, 0)
						&& windowImage_ij.at<float>(1, 1)
								> windowImage_ij.at<float>(0, 2)) {
					thresh.at<uchar>(i + 1, j + 1) = 255;
					D_new.at<float>(i + 1, j + 1) = windowImage_ij.at<float>(1,
							1);
				}
			}

		}
	}

	//Step 4: Hysteresis Thresholding
	Scalar meanValue = mean(D, thresh);
	float thigh = meanValue[0];
	float tlow = thigh / 2;

	for (int i = 0; i < rowBlocks3x; i++) {
		for (int j = 0; j < colBlocks3x; j++) {
			Mat windowImage_ij(D_new, Rect(j, i, BLOCK_SIZE3X, BLOCK_SIZE3X));
			Mat windowImage5x_ij;
			if (rowBlocks3x + BLOCK_SIZE < imgGray.rows
					&& colBlocks3x + BLOCK_SIZE < imgGray.cols) {
				Mat windowImage5x_ij(D_new, Rect(j, i, BLOCK_SIZE, BLOCK_SIZE));
			}

			// block processing loops
			bool hasHighValue = false;
			bool hasAtLeastOneAverageValue = false;
			if (windowImage_ij.at<float>(1, 1) > thigh) {
				dest.at<uchar>(i + 1, j + 1) = 255;

			} else if (windowImage_ij.at<float>(1, 1) < tlow) {
				continue;
			} else {
				for (int k = 0; k < windowImage_ij.rows; k++) {
					for (int m = 0; m < windowImage_ij.cols; m++) {
						if (k == 1 && m == 1) {
							continue;
						}
						if (windowImage_ij.at<float>(k, m) > thigh) {
							hasHighValue = true;
							break;
						} else if (windowImage_ij.at<float>(k, m) > tlow) {
							hasAtLeastOneAverageValue = true;
						}
					}
				}
				if (hasHighValue) {
					dest.at<uchar>(i + 1, j + 1) = 255;
				} else if (hasAtLeastOneAverageValue
						&& rowBlocks3x + BLOCK_SIZE < imgGray.rows
						&& colBlocks3x + BLOCK_SIZE < imgGray.cols) {
					for (int k = 0; k < windowImage5x_ij.rows; k++) {
						for (int m = 0; m < windowImage5x_ij.cols; m++) {
							if (k == 2 && m == 2) {
								continue;
							}

							if (windowImage5x_ij.at<float>(k, m) > thigh) {
								hasHighValue = true;
								break;
							}

						}
					}

					if (hasHighValue) {
						dest.at<uchar>(i + 1, j + 1) = 255;
					}

				}
			}

		}
	}

}



