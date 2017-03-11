#include <iostream>
#include <fstream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

Mat img, imgg, imgn, imgt, imer, imdil, color;
double thresh = 80;
double maxValue = 255;

CvScalar white = CV_RGB(255,255,255);
CvScalar green = CV_RGB(0,255,0);

int erosion_size=2;
int dilation_size=1;

Mat elm1 = getStructuringElement(MORPH_CROSS, Size(2*erosion_size + 1, 2*erosion_size+  1), Point(erosion_size, erosion_size));
Mat elm2 = getStructuringElement(MORPH_CROSS, Size(2*dilation_size + 1, 2*dilation_size+  1), Point(dilation_size, dilation_size));

void thinningIteration(cv::Mat& im, int iter)
{
    cv::Mat marker = cv::Mat::zeros(im.size(), CV_8UC1);

    for (int i = 1; i < im.rows-1; i++)
    {
        for (int j = 1; j < im.cols-1; j++)
        {
            uchar p2 = im.at<uchar>(i-1, j);
            uchar p3 = im.at<uchar>(i-1, j+1);
            uchar p4 = im.at<uchar>(i, j+1);
            uchar p5 = im.at<uchar>(i+1, j+1);
            uchar p6 = im.at<uchar>(i+1, j);
            uchar p7 = im.at<uchar>(i+1, j-1);
            uchar p8 = im.at<uchar>(i, j-1);
            uchar p9 = im.at<uchar>(i-1, j-1);

            int A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) + 
                     (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) + 
                     (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                     (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
            int B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
            int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
            int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

            if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
                marker.at<uchar>(i,j) = 1;
        }
    }

    im &= ~marker;
}

/*Fungsi Skeletoning*/
void thinning(cv::Mat& im)
{
    im /= 255;

    cv::Mat prev = cv::Mat::zeros(im.size(), CV_8UC1);
    cv::Mat diff;

    do {
        thinningIteration(im, 0);
        thinningIteration(im, 1);
        cv::absdiff(im, prev, diff);
        im.copyTo(prev);
    } 
    while (cv::countNonZero(diff) > 0);

    im *= 255;
}

/*Program Ekstraksi Minutiae*/
int main(int argc, char* argv[])
{
	Mat img = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	cvtColor(img,color,CV_GRAY2BGR);
	if (img.empty())
	{
		cout << "Unable to load image" << endl;
		system("pause");
		return -1;
	}
	namedWindow("thin", CV_WINDOW_NORMAL);
	namedWindow("gambar", CV_WINDOW_NORMAL);
	namedWindow("hasil", CV_WINDOW_NORMAL);
	
	cv::threshold(img, imgt, thresh, maxValue, CV_THRESH_BINARY_INV);

	thinning(imgt);
	imshow("thin", imgt);
	int a=0 ,b=0 ,c=0 ,d=0 ,e=0 ,f=0 ,g=0 ,h=0 ,i=0;
	int ridge = 0, ridcheck = 0;
	int bif = 0, bifcheck =0;

	
	for(int y=0; y<imgt.rows;y++)
	{
		for(int x=0; x<imgt.cols;x++)	
		{
			int pix1 = imgt.at<uchar>(x-1,y-1);
			int pix2 = imgt.at<uchar>(x,y-1);
			int pix3 = imgt.at<uchar>(x+1,y-1);
			int pix4 = imgt.at<uchar>(x-1,y);
			int pix5 = imgt.at<uchar>(x,y);
			int pix6 = imgt.at<uchar>(x+1,y);
			int pix7 = imgt.at<uchar>(x-1,y+1);
			int pix8 = imgt.at<uchar>(x,y+1);
			int pix9 = imgt.at<uchar>(x+1,y+1);
			
			if(pix1 > 80) a=1;
			if(pix2 > 80) b=1;
			if(pix3 > 80) c=1;
			if(pix4 > 80) d=1;
			if(pix5 > 80) e=1;
			if(pix6 > 80) f=1;
			if(pix7 > 80) g=1;
			if(pix8 > 80) h=1;
			if(pix9 > 80) i=1;
			
			if(a==1 && b==0 && c==0 && d==0 && e==1 && f==0 && g==0 && h==0 && i==0) ridge++, ridcheck++;
			if(a==0 && b==1 && c==0 && d==0 && e==1 && f==0 && g==0 && h==0 && i==0) ridge++, ridcheck++;
			if(a==0 && b==0 && c==1 && d==0 && e==1 && f==0 && g==0 && h==0 && i==0) ridge++, ridcheck++;
			if(a==0 && b==0 && c==0 && d==1 && e==1 && f==0 && g==0 && h==0 && i==0) ridge++, ridcheck++;
			if(a==0 && b==0 && c==0 && d==0 && e==1 && f==1 && g==0 && h==0 && i==0) ridge++, ridcheck++;
			if(a==0 && b==0 && c==0 && d==0 && e==1 && f==0 && g==1 && h==0 && i==0) ridge++, ridcheck++;
			if(a==0 && b==0 && c==0 && d==0 && e==1 && f==0 && g==0 && h==1 && i==0) ridge++, ridcheck++;
			if(a==0 && b==0 && c==0 && d==0 && e==1 && f==0 && g==0 && h==0 && i==1) ridge++, ridcheck++;
			
			if(a==1 && b==0 && c==0 && d==0 && e==1 && f==1 && g==1 && h==0 && i==0) bif++, bifcheck++;
			if(a==0 && b==1 && c==0 && d==1 && e==1 && f==0 && g==0 && h==0 && i==1) bif++, bifcheck++;
			if(a==1 && b==0 && c==1 && d==0 && e==1 && f==0 && g==0 && h==1 && i==0) bif++, bifcheck++;
			if(a==0 && b==1 && c==0 && d==0 && e==1 && f==1 && g==1 && h==0 && i==0) bif++, bifcheck++;
			if(a==0 && b==0 && c==1 && d==1 && e==1 && f==0 && g==0 && h==0 && i==1) bif++, bifcheck++;
			if(a==1 && b==0 && c==0 && d==0 && e==1 && f==1 && g==0 && h==1 && i==0) bif++, bifcheck++;
			if(a==0 && b==1 && c==0 && d==0 && e==1 && f==0 && g==1 && h==0 && i==1) bif++, bifcheck++;
			if(a==0 && b==0 && c==1 && d==1 && e==1 && f==0 && g==0 && h==1 && i==0) bif++, bifcheck++;
			//if(b==0 && c==1 && e==1 && f==1 && h==0 && i==1) bif++, bifcheck++;
			//if(a==1 && b==0 && h==0 && e==1 && d==1 && g==1) bif++, bifcheck++;
			//if(a==1 && b==1 && c==1 && e==1 && g==0 && i==1) bif++, bifcheck++;
			//if(d==0 && f==0 && e==1 && g==1 && h==1 && i==1) bif++, bifcheck++;
			
			if((ridge))
			{	
				Point pt = Point(y,x);
				circle(color, pt, 4, green, 1, 8, 0);
				circle(imgt, pt, 4, white, 1, 8, 0);
				ridge = 0;
			}
			
			if((bif))
			{	
				Point pt1 = Point(y-2,x-2);
				Point pt2 = Point(y+2,x+2);
				rectangle(color, pt1, pt2, green, 1, 8, 0);
				rectangle(imgt, pt1, pt2, white, 1, 8, 0);
				bif = 0;
			}
			
			a=0,b=0,c=0,d=0,e=0,f=0,g=0,h=0,i=0;			
		}
	}
	imshow("gambar", imgt);
	imshow("hasil", color);
	cout << "jumlah ridge : " << ridcheck << endl;
	cout << "jumlah bif : " << bifcheck << endl;

	if(waitKey(0)==27)
	return 0;
}