
#include <iostream>
#include <vector>
#include <math.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/opencv.hpp"
#include <algorithm>
#include <cstdarg>

using namespace std;
using namespace cv;


Mat get_indices(Mat M, int cnt)
{
   
    Mat ind; 
    for(int i = 0; i < M.rows; i++)
    {
        for(int j = 0; j < M.cols; j++)
        {
         if(M.at<uchar>(i,j) != 0)
         {
            Mat y  = Mat::zeros(1,2,CV_32SC1);
            y.at<int>(0,0) = i;
            y.at<int>(0,1) = j;

            ind.push_back(y);
                       
            
         }
        }    
    }

    return ind;
}

int search_ind(Mat idx,int i, int j)
{
    for(int k = 0; k < idx.rows; k++)
    {
      if( (idx.at<int>(k,0) == i) && (idx.at<int>(k,1) == j) )
      {
          return k;
      }
    }

    
        return -1;
    
}

Mat get_img(Mat X, Mat idx, int R, int C)
{
    Mat I = Mat::zeros(R,C,CV_32F);
    for(int i = 0; i < idx.rows; i++)
    {
        I.at<float>(idx.at<int>(i,0),idx.at<int>(i,1)) = X.at<float>(i,0);
    }
    

    I.convertTo(I,CV_8UC1);
    return I;
}

Mat do_poisson(Mat ss, Mat tt, Mat M)
{
   
M.convertTo(M,CV_8UC1);
normalize(M,M,0,1,NORM_MINMAX);

Mat k = Mat::zeros(3,3,CV_8U);

            k.at<uchar>(0,1) = 1;
k.at<uchar>(1,0) = 1;k.at<uchar>(1,1) = 1;k.at<uchar>(1,2) = 1;
            k.at<uchar>(2,1) = 1;

M.setTo(1,M);
dilate(M,M,k);                      /*Mask region defined as omega + small dilation on omega*/
M.setTo(1,M);


Mat M1,M2;
erode(M,M1,k);

M1.setTo(1,M1);

// M2 = M2 - M;
M1 += M;


/*Do for all layers*/

vector<Mat> ss1;
vector<Mat> tt1;

split(ss,ss1);
split(tt,tt1);

for(int iin = 0; iin < 3; iin++)
{
     Mat S,T,b,x;

     S = ss1[iin];
     T = tt1[iin];
    
int cnt = countNonZero(M1);
b = Mat::zeros(cnt,1,CV_32F);
int n[] = {cnt,cnt};
SparseMat A(2,n,CV_32FC1);

Mat idx = get_indices(M1,cnt);
                                                                                                                                                                                                                                                                
for(int i = 0; i < M.rows; i++)             /*Generate A*/
{
    for(int j = 0; j < M.cols; j++)
    {
            
        if(M1.at<uchar>(i,j) == 2)
        {
            int ind = search_ind(idx,i,j);
            n[0] = ind; n[1] = ind;
            A.ref<float>(n) = -4;
//             b.at<float>(ind,0) += -4 * S.at<uchar>(i,j);                        // remove for inpainting Set Del^2 [I] = 0
            
            for(int k = -1; k <= 1; k++)            
            {
                for(int l = -1; l <= 1; l++)
                {
                    if(( !(k == -1 && l == 0) && !(k == 0 && l == -1) && !(k == 1 && l == 0) && !(k == 0 && l == 1) ) )
                    {
                        continue;
                    }
                

                    n[1] = search_ind(idx,i+k,j+l);
                    A.ref<float>(n) = 1;
                    
//                     b.at<float>(ind,0) += S.at<uchar>(i+k,j+l);                        // remove for inpainting Set Del^2 [I] = 0
                    
                }

            }
                    
            
        }
        else if(M1.at<uchar>(i,j) == 1)
        {
            int ind = search_ind(idx,i,j);
            n[0] = ind; n[1] = ind;
            A.ref<float>(n) = -4;
//             b.at<float>(ind,0) += -4 * S.at<uchar>(i,j);                        // remove for inpainting Set Del^2 [I] = 0
            
            for(int k = -1; k <= 1; k++)            
            {
                for(int l = -1; l <= 1; l++)
                {
                    if(( !(k == -1 && l == 0) && !(k == 0 && l == -1) && !(k == 1 && l == 0) && !(k == 0 && l == 1) ) )
                    {
                        continue;
                    }
                    
                    if(M1.at<uchar>(i+k,j+l) == 0)   /*T Value*/
                    {
                        b.at<float>(ind,0) -= T.at<uchar>(i+k,j+l);
                    }
                    else                            /*I Ceoffs*/
                    {

                        n[1] = search_ind(idx,i+k,j+l);
                        A.ref<float>(n) = 1;
                    }
                    
//                     b.at<float>(ind,0) += S.at<uchar>(i+k,j+l);                     // remove for inpainting Set Del^2 [I] = 0
                }

            }
                    
            
        }
        
            
        
        
    }
    
}

Mat Y; A.convertTo(Y,CV_32FC1);

solve(Y,b,x,DECOMP_NORMAL);

A.release();
Y.release();

Mat U = get_img(x,idx,M.rows,M.cols);
U.copyTo(T,U);

tt1[iin] = T;

// cout << "\nDone "<<iin<<"\n";
}

Mat TT;
merge(tt1,TT);

tt1.clear();
ss1.clear();
// imwrite("Inpainted.jpg",TT);

return TT;

}


int main()
{


// VideoCapture cap;
// cap.open("greenvid.MOV");

int C = 1920 / 2, R = 1080 / 2;
    
VideoWriter wir;
int codec = CV_FOURCC('M','J','P','G');
wir.open("Result_GS.avi",codec ,25,Size( C,R ));

VideoWriter wir2;
wir2.open("GS_Original.avi",codec ,25,Size( C,R ));

int i = 0;

Mat S,M,P;

string imgpath("/home/hulio/RPI/CVFX/greenvid/");
string mask("/home/hulio/RPI/CVFX/Mask/");

while(i <= 100)
{


    string img1 = imgpath;
    string mask1 = mask;
    
    stringstream convert;
    convert << i;
    img1.append("greenvid");
    img1.append(convert.str());
    mask1.append("M");
    mask1.append(convert.str());
    img1.append(".jpeg");
    mask1.append(".jpg");

    
    S = imread(img1.data());
    M = imread(mask1.data(),0);

    resize(S,S,Size(C,R));
    resize(M,M,Size(C,R));

    
//     cout << img1.data() << "\n";
//     cout << mask1.data() << "\n";
//     cout << i << "\n";
    
    P = do_poisson(S,S,M);

    wir << P;
    wir2 << S;
    cout << i << "\n";
    i++;
    

}

wir.release();
wir2.release();

return 1;
}
