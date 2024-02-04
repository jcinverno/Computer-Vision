__kernel void sobel_threshold(__global uchar* image, int w, int h, int padding, __global uchar* imageOut, float t1, float t2) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int idx = y * (w * 3 + padding) + x * 3;

    if ((x < w) && (y < h)) {

        float diff = (image[idx] - image[idx + 1]) + (image[idx] - image[idx + 2]) + (image[idx + 1] - image[idx + 2]);
        float average = (image[idx] + image[idx + 1] + image[idx +2]) / 3.0;

        if (diff < t1 && average > t2) {
            imageOut[idx] = convert_uchar_sat(255);
            imageOut[idx+1] = convert_uchar_sat(255);
            imageOut[idx+2] = convert_uchar_sat(255);
        } else {
            imageOut[idx] = convert_uchar_sat(0);
            imageOut[idx+1] = convert_uchar_sat(0);
            imageOut[idx+2] = convert_uchar_sat(0);
        }
    }
}
