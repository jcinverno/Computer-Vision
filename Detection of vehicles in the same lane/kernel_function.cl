#define MAX_RHO 10000

// Sampler
__constant sampler_t sampler =
    CLK_NORMALIZED_COORDS_FALSE | // Natural coordinates
    CLK_ADDRESS_CLAMP_TO_EDGE |  // Clamp to zeros
    CLK_FILTER_NEAREST;

// Function to calculate Sobel filter
void calculateSobel(__global uchar4* global_image, int x, int y, int width, uchar* G) {
    int2 offsets[] = {
        (int2)(-1, -1),
        (int2)(0, -1),
        (int2)(1, -1),
        (int2)(-1, 0),
        (int2)(1, 0),
        (int2)(-1, 1),
        (int2)(0, 1),
        (int2)(1, 1),
    };

    int Gx = 0;

    for (int i = 0; i < 8; i++) {
        int2 offset = offsets[i];
        int index = (y + offset.y) * width + (x + offset.x);
        uchar4 pixel = global_image[index];

        Gx += offset.x * pixel.x;
    }

    *G = sqrt((float)(Gx * Gx));
}


// Function for Hough Transform
void HoughTransform(uchar4 pixel, __global int* accumulator_left,
                    __global int* accumulator_right, int width, int height, int x, int y, int L_Min_Ang, int L_Max_Ang, int R_Min_Ang, int R_Max_Ang) {

    if (pixel.x == 255 && pixel.y == 255 && pixel.z == 255) {
        if(x <= 600){
            // Iterate through angles for left lines
            for (int i = 30; i < 50; i++) {
                double theta = i * M_PI / 180;
                int rho = (int)(x * cos(theta) + y * sin(theta));
                if (rho > 0 && rho < MAX_RHO){
                    int accumulatorLeftIdx = rho * 180 + i;
                    atomic_add(&accumulator_left[accumulatorLeftIdx], 1);
                }
            }
        }
        if(x > 600){

            // Iterate through angles for right lines
            for (int a = 135; a < 155; a++) {
                double theta1 = a * M_PI / 180;
                int rho1 = (int)(x * cos(theta1) + y * sin(theta1) + 948/2);
                if (rho1 > 0 && rho1 < MAX_RHO){
                    int accumulatorRightIdx = rho1 * 180 + a;
                    atomic_add(&accumulator_right[accumulatorRightIdx], 1);
                }
            }
        }
    }
}



// Function to determine Region Of Interest
int isInsideROI(int x, int y, int width, int height, __global int2* roi_vertices, int num_vertices) {
    int i, j;
    int inside = 0;
    for (i = 0, j = num_vertices - 1; i < num_vertices; j = i++) {
        if (((roi_vertices[i][1] > y) != (roi_vertices[j][1] > y)) &&
            (x < (roi_vertices[j][0] - roi_vertices[i][0]) * (y - roi_vertices[i][1]) / (roi_vertices[j][1] - roi_vertices[i][1]) + roi_vertices[i][0])) {
            inside = !inside;
        }
    }
    return inside;
}


__kernel void KernelFunction(__global uchar4* inputImage, __global uchar4* outputImage, __global int* accumulator_left,
                             __global int* accumulator_right, const int width, const int height,const int t1,const int t2,
                             __global int2* roi_vertices,const int num_vertices,const int L_Min_Ang,const int L_Max_Ang,
                             const int R_Min_Ang, const int R_Max_Ang) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    int G;
    int Gy;

    uchar4 Test_Image = outputImage[y * width + x];

    if (isInsideROI(x, y, width, height, roi_vertices, num_vertices)) {

        uchar4 pixel = inputImage[y * width + x];

        if (pixel.x > t2 && pixel.y > t2 && pixel.z > t2) {

            calculateSobel(inputImage, x, y, width, &G);

            if (G > t1) {
                Test_Image = (uchar4)(255, 255, 255, 0);
                if (x < 500 || x > 700) {
                    HoughTransform(Test_Image, accumulator_left, accumulator_right, width, height, x, y, L_Min_Ang, L_Max_Ang, R_Min_Ang, R_Max_Ang);
                }
            }
        } else {
            Test_Image = (uchar4)(0, 0, 0, 0);
        }
    }
    outputImage[y * width + x] = Test_Image;
}