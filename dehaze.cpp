#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

/* 
 * Dark Channel Prior
 * J^dark(x) = min(min(J^c(y))), c¡Ê{R, G, B}
 */
Mat Dark_Channel(Mat src, int height, int width, int ksize) 
{
	Mat dst(height, width, CV_32FC1, Scalar(0)),
		drk(height, width, CV_32FC1, Scalar(0));
	
	for (int i = 0; i < height; i++) 
	{
		Vec<float, 3>* p = src.ptr<Vec<float, 3>>(i);
		Vec<float, 1>* q = drk.ptr<Vec<float, 1>>(i);
		for (int j = 0; j < width; j++) 
		{
			q[j] = min(
				min(p[j][0], p[j][1]),
				min(p[j][0], p[j][2]));
		}
	}
	
	erode(drk, dst, Mat::ones(ksize, ksize, CV_32FC1));
		
	return dst;
}

/*
 * Atmospheric Light
 * A^c = ¡ÆTOP^c(x) / N, c¡Ê{R, G, B}
 */
Vec<float, 3> Air_Light(Mat src, Mat dark, int height, int width, float ratio) 
{
	int N = ratio * height * width;
	Mat dark_in = dark.reshape(1, height*width);

	vector<int> max_index;
	float max_num = 0;

	Vec<float, 3> airlight(0, 0, 0);
	Mat top_pixels = Mat::ones(N, 1, CV_32FC3);

	for (int i = 0; i < N; i++)
	{
		max_num = 0;
		max_index.push_back(max_num);

		for (float* p = (float*)dark_in.datastart; p != (float*)dark_in.dataend; p++) 
		{
			if (*p > max_num) 
			{
				max_num = *p;
				max_index[i] = (p - (float*)dark_in.datastart);		// Get index of a bright pixel
				top_pixels.at<Vec<float, 3>>(i, 0) = ((Vec<float, 3>*)src.data)[max_index[i]];
			}
		}
		((float*)dark_in.data)[max_index[i]] = 0;					// Avoid repetitive visiting
	}

	for (int i = 0; i < N; i++) 
	{
		Vec<float, 3>* ptr = top_pixels.ptr<Vec<float, 3>>(i);
		airlight[0] += ptr[0][0];
		airlight[1] += ptr[0][1];
		airlight[2] += ptr[0][2];
	}

	airlight[0] /= N;
	airlight[1] /= N;
	airlight[2] /= N;

	return airlight;
}

/*
 * Get Transmissivity Image
 * t(x) = 1 - w * d
 */
Mat Transmissivity_Image(Mat src, int height, int width, Vec<float, 3> A, float w, int ksize)
{
	float ave = (A[0] + A[1] + A[2]) / 3.0;

	Mat drk(height, width, CV_32FC1, Scalar(0));
	Mat dst(height, width, CV_32FC1, Scalar(0));

	for (int i = 0; i < height; i++) 
	{
		Vec<float, 3>* p = src.ptr<Vec<float, 3>>(i);
		Vec<float, 1>* q = drk.ptr<Vec<float, 1>>(i);
		for (int j = 0; j < width; j++) 
		{
			q[j] = min(
				min(p[j][0] / ave, p[j][1] / ave),
				min(p[j][0] / ave, p[j][2] / ave));
		}
	}
	erode(drk, dst, Mat::ones(ksize, ksize, CV_32FC1));

	for (int i = 0; i < height; i++)
	{
		float* ptr = dst.ptr<float>(i);
		for (int j = 0; j < width; j++)
		{
			ptr[j] = 1.0 - w * ptr[j];
			ptr[j] = ptr[j] >= 0.0 ? ptr[j] : 0.0;
			ptr[j] = ptr[j] <= 1.0 ? ptr[j] : 1.0;
		}
	}
	return dst;
}

Mat Guided_Filter(Mat src, Mat p, int height, int width, int fsize, double eps) {
	Mat i(height, width, CV_32FC1, Scalar(0));
	cvtColor(src, i, CV_BGR2GRAY);
	
	i /= 255.0;
	
	Mat mean_I;
	boxFilter(i, mean_I, CV_32FC1, Size(fsize, fsize));
	Mat mean_P;
	boxFilter(p, mean_P, CV_32FC1, Size(fsize, fsize));
	Mat corr_II;
	boxFilter(i.mul(i), corr_II, CV_32FC1, Size(fsize, fsize));
	Mat corr_IP;
	boxFilter(i.mul(p), corr_IP, CV_32FC1, Size(fsize, fsize));

	Mat var_I = corr_II - mean_I.mul(mean_I);
	Mat cov_IP = corr_IP - mean_I.mul(mean_P);

	Mat a = cov_IP / (var_I + eps);
	Mat b = mean_P - a.mul(mean_I);

	Mat mean_a;
	boxFilter(a, mean_a, CV_32FC1, Size(fsize, fsize));
	Mat mean_b;
	boxFilter(b, mean_b, CV_32FC1, Size(fsize, fsize));

	Mat q = mean_a.mul(i) + mean_b;
	return q;
}

Mat Dehaze(Mat src, Mat t, int height, int width, Vec<float, 3> A, float t0, float exposure) {
	float a = A[0];
	a = a >= A[1] ? a : A[1];
	a = a >= A[2] ? a : A[2];
	
	Vec<float, 3>* p = (Vec<float, 3>*)src.datastart;
	float* q = (float*)t.datastart;

	for (; p < (Vec<float, 3>*)src.dataend && q < (float*)t.dataend; p++, q++) {
		(*p)[0] = ((*p)[0] - A[0]) / std::max(*q, t0) + A[0] + exposure;
		(*p)[1] = ((*p)[1] - A[1]) / std::max(*q, t0) + A[1] + exposure;
		(*p)[2] = ((*p)[2] - A[2]) / std::max(*q, t0) + A[2] + exposure;
	}
	return src;
}

int main() 
{
	string in_path = "./Dataset/haze/";		// Input path
	string out_path = "./Dehaze/dehaze/";	// Output path
	string img_name = "haze10";				// Image name
	
	Mat src = imread(in_path + img_name + ".jpg");
	if (!src.data) 
	{
		cerr << "Reading Image Failed!" << endl;
		return -1;
	}
	src.convertTo(src, CV_32FC3);	// Convert CV_8UC3 to CV_32FC3
	  
	int height = src.rows;		// Image height
	int width = src.cols;		// Image width
	int ksize = 31;				// Minimum filter size
	int fsize = 121;			// Guided filter size
	Mat dark;					// Dark channel of image
	Mat coarse_t;				// Coarse tranmissivity image
	Mat refine_t;				// Refined transmissivity image
	Vec<float, 3> A;			// Atmospheric light: A

	float ratio = 0.001;		// Ratio of picking up brightest pixels in image
	float w = 0.95;				// Preserve a little haze to indicate depth
	float eps = 0.001;			// Epsilon
	float t0 = 0.1;				// Minimum value of t(x£©
	float exposure = 0;			// Exposure compensation

	double t1, t2, t3, t4;

	t1 = getTickCount();

	dark = Dark_Channel(src, height, width, ksize);	// Get dark channel of image

	t2 = getTickCount();
	cout << "Get dark channel: " << (t2 - t1) / getTickFrequency() * 1000 << "ms" << endl;

	A = Air_Light(src, dark, height, width, ratio);			// Get atmospheric light value						
	cout << "Air Light: B:" << A[0] << ", G:" << A[1] << ", R:" << A[2] << endl;
	
	t3 = getTickCount();
	cout << "Get air light: " << (t3 - t2) / getTickFrequency() * 1000 << "ms" << endl;

	coarse_t = Transmissivity_Image(src, height, width, A, w, ksize);		// Get coarse transmissivity image

	t4 = getTickCount();
	cout << "Get coarse t(x): " << (t4 - t3) / getTickFrequency() * 1000 << "ms" << endl;

	refine_t = Guided_Filter(src, coarse_t, height, width, fsize, eps);	// Get refined transmissivity image

	t2 = getTickCount();
	cout << "Get refined t(x): " << (t2 - t4) / getTickFrequency() * 1000 << "ms" << endl;
	
	Mat dehazed = Dehaze(src, refine_t, height, width, A, t0, exposure); // Get dehazed image
	
	t3 = getTickCount();
	cout << "Get dehazed: " << (t3 - t2) / getTickFrequency() * 1000 << "ms" << endl;

	t4 = getTickCount();
	cout << "Total: " << (t4 - t1) / getTickFrequency() * 1000 << "ms" << endl;

	//imwrite(out_path + img_name + "_t(x)_2.jpg", refine_t * 255);
	//imwrite(out_path + img_name + "_dehaze_2.jpg", dehazed);

	return 0;
}
