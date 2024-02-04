using Emgu.CV;
using Emgu.CV.Structure;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Reflection;
using System.Windows.Forms;

namespace SS_OpenCV
{
    class ImageClass
    {

        /// <summary>
        ///Direct access to memory - faster method
        /// </summary>
        /// <param name="img">Image</param>
        public static void Negative(Image<Bgr, byte> img)
        {
            unsafe
            {
                // direct access to the image memory(sequencial)
                // direcion top left -> bottom right

                MIplImage m = img.MIplImage;
                byte* dataPtr = (byte*)m.ImageData.ToPointer(); // Pointer to the image
                byte blue, green, red;

                int width = img.Width;
                int height = img.Height;
                int nChan = m.NChannels; // number of channels - 3
                int padding = m.WidthStep - m.NChannels * m.Width; // alinhament bytes (padding)
                int x, y;

                if (nChan == 3) // image in RGB
                {
                    for (y = 0; y < height; y++)
                    {
                        for (x = 0; x < width; x++)
                        {
                            //retrieve 3 colour components
                            blue = dataPtr[0];
                            green = dataPtr[1];
                            red = dataPtr[2];

                            // store in the image
                            dataPtr[0] = (byte)(255 - blue);
                            dataPtr[1] = (byte)(255 - green);
                            dataPtr[2] = (byte)(255 - red);

                            // advance the pointer to the next pixel
                            dataPtr += nChan;
                        }

                        //at the end of the line advance the pointer by the aligment bytes (padding)
                        dataPtr += padding;
                    }
                }
            }
        }

        /// <summary>
        /// Convert to gray
        /// Direct access to memory - faster method
        /// </summary>
        /// <param name="img">image</param>
        public static void ConvertToGray(Image<Bgr, byte> img)
        {
            unsafe
            {
                // direct access to the image memory(sequencial)
                // direcion top left -> bottom right

                MIplImage m = img.MIplImage;
                byte* dataPtr = (byte*)m.ImageData.ToPointer(); // Pointer to the image
                byte blue, green, red, gray;

                int width = img.Width;
                int height = img.Height;
                int nChan = m.NChannels; // number of channels - 3
                int padding = m.WidthStep - m.NChannels * m.Width; // alinhament bytes (padding)
                int x, y;

                if (nChan == 3) // image in RGB
                {
                    for (y = 0; y < height; y++)
                    {
                        for (x = 0; x < width; x++)
                        {
                            //retrieve 3 colour components
                            blue = dataPtr[0];
                            green = dataPtr[1];
                            red = dataPtr[2];

                            // convert to gray
                            gray = (byte)Math.Round(((int)blue + green + red) / 3.0);

                            // store in the image
                            dataPtr[0] = gray;
                            dataPtr[1] = gray;
                            dataPtr[2] = gray;

                            // advance the pointer to the next pixel
                            dataPtr += nChan;
                        }

                        //at the end of the line advance the pointer by the aligment bytes (padding)
                        dataPtr += padding;
                    }
                }
            }
        }

        /// <summary>
        /// Brightness and Contrast
        /// Direct access to memory - faster method
        /// </summary>
        /// <param name="img">Image</param>
        public static void BrightContrast(Image<Bgr, byte> img, int bright, double contrast)
        {
            unsafe
            {
                // direct access to the image memory(sequencial)
                // direcion top left -> bottom right

                MIplImage m = img.MIplImage;
                byte* dataPtr = (byte*)m.ImageData.ToPointer(); // Pointer to the image

                int width = img.Width;
                int height = img.Height;
                int nChan = m.NChannels; // number of channels - 3
                int padding = m.WidthStep - m.NChannels * m.Width; // alinhament bytes (padding)
                int x, y;
                int blue, green, red;

                if (nChan == 3) // image in RGB
                {
                    for (y = 0; y < height; y++)
                    {
                        for (x = 0; x < width; x++)
                        {
                            //retrieve 3 colour components
                            blue = dataPtr[0];
                            green = dataPtr[1];
                            red = dataPtr[2];

                            blue = Convert.ToInt32((contrast * blue) + bright);
                            green = Convert.ToInt32((contrast * green) + bright);
                            red = Convert.ToInt32((contrast * red) + bright);


                            if (blue < 0) blue = 0;
                            if (green < 0) green = 0;
                            if (red < 0) red = 0;

                            if (blue > 255) blue = 255;
                            if (green > 255) green = 255;
                            if (red > 255) red = 255;

                            dataPtr[0] = (byte)blue;
                            dataPtr[1] = (byte)green;
                            dataPtr[2] = (byte)red;


                            // advance the pointer to the next pixel
                            dataPtr += nChan;
                        }

                        //at the end of the line advance the pointer by the aligment bytes (padding)
                        dataPtr += padding;
                    }
                }
            }
        }

        /// <summary>
        /// Image component
        /// Direct access to memory - faster method
        /// </summary>
        /// <param name="img">Image</param>
        public static void RedChannel(Image<Bgr, byte> img)
        {
            unsafe
            {
                // direct access to the image memory(sequencial)
                // direcion top left -> bottom right

                MIplImage m = img.MIplImage;
                byte* dataPtr = (byte*)m.ImageData.ToPointer(); // Pointer to the image
                byte blue, green, red;

                int width = img.Width;
                int height = img.Height;
                int nChan = m.NChannels; // number of channels - 3
                int padding = m.WidthStep - m.NChannels * m.Width; // alinhament bytes (padding)
                int x, y;

                if (nChan == 3) // image in RGB
                {
                    for (y = 0; y < height; y++)
                    {
                        for (x = 0; x < width; x++)
                        {
                            //retrieve 3 colour components
                            blue = dataPtr[0];
                            green = dataPtr[1];
                            red = dataPtr[2];

                            // store in the image
                            dataPtr[0] = red;
                            dataPtr[1] = red;
                            dataPtr[2] = red;

                            // advance the pointer to the next pixel
                            dataPtr += nChan;
                        }

                        //at the end of the line advance the pointer by the aligment bytes (padding)
                        dataPtr += padding;
                    }
                }
            }
        }

        /// <summary>
        /// Rotation
        /// Direct access to memory - faster method
        /// </summary>
        /// <param name="img">Image</param>
        public static void Rotation(Image<Bgr, byte> imgCopy, Image<Bgr, byte> imgOrigem, float angle)
        {
            unsafe
            {
                // direct access to the image memory(sequencial)
                // direcion top left -> bottom right

                MIplImage origem = imgOrigem.MIplImage;
                byte* dataPtr_o = (byte*)origem.ImageData.ToPointer(); // Pointer to the image

                MIplImage copy = imgCopy.MIplImage;
                byte* dataPtr_c = (byte*)copy.ImageData.ToPointer(); // Pointer to the image

                int width = origem.Width;
                int height = origem.Height;
                int nChan = origem.NChannels; // number of channels - 3
                int padding = origem.WidthStep - origem.NChannels * origem.Width; // alinhament bytes (padding)
                int x, y, xo, yo;
                int widthstep = origem.WidthStep;
                byte blue, green, red;

                double H = height / 2.0;
                double W = width / 2.0;
                double cos = Math.Cos(angle);
                double sin = Math.Sin(angle);

                if (nChan == 3) // image in RGB
                {
                    for (y = 0; y < height; y++)
                    {
                        for (x = 0; x < width; x++)
                        {
                            //retrieve 3 color componets from copy
                            yo = (int)(Math.Round(H - (x - W) * sin - (H - y) * cos));
                            xo = (int)(Math.Round((x - W) * cos - (H - y) * sin + W));


                            if ((xo < width && xo >= 0) && (yo < height && yo >= 0))
                            {
                                //retrieve 3 colour components from origem
                                blue = (byte)(dataPtr_o + yo * widthstep + xo * nChan)[0];
                                green = (byte)(dataPtr_o + yo * widthstep + xo * nChan)[1];
                                red = (byte)(dataPtr_o + yo * widthstep + xo * nChan)[2];

                                dataPtr_c[0] = blue;
                                dataPtr_c[1] = green;
                                dataPtr_c[2] = red;

                            }
                            else
                            {
                                dataPtr_c[0] = 0;
                                dataPtr_c[1] = 0;
                                dataPtr_c[2] = 0;
                            }
                            // advance the pointer to the next pixel
                            dataPtr_c += nChan;
                        }

                        //at the end of the line advance the pointer by the aligment bytes (padding)
                        dataPtr_c += padding;
                    }
                }
            }
        }

        /// <summary>
        /// Translation
        /// </summary>
        /// <param name="img">Image</param>
        public static void Translation(Image<Bgr, byte> imgDestino, Image<Bgr, byte> imgOrigem, int dx, int dy)

        {
            unsafe
            {
                // direct access to the image memory(sequencial)
                // direcion top left -> bottom right

                MIplImage origem = imgOrigem.MIplImage;
                byte* dataPtr_o = (byte*)origem.ImageData.ToPointer(); // Pointer to the image

                MIplImage copy = imgDestino.MIplImage;
                byte* dataPtr_c = (byte*)copy.ImageData.ToPointer(); // Pointer to the image

                int width = origem.Width;
                int height = origem.Height;
                int nChan = origem.NChannels; // number of channels - 3
                int padding = origem.WidthStep - origem.NChannels * origem.Width; // alinhament bytes (padding)
                int x, y, xo, yo;
                int widthstep = origem.WidthStep;
                byte blue, green, red;

                if (nChan == 3) // image in RGB
                {
                    for (y = 0; y < height; y++)
                    {
                        for (x = 0; x < width; x++)
                        {
                            //coordinates from the image origin
                            yo = (y - dy);
                            xo = (x - dx);


                            if ((xo < width && xo >= 0) && (yo < height && yo >= 0))
                            {
                                //retrieve 3 colour components from image origin 
                                blue = (byte)(dataPtr_o + yo * widthstep + xo * nChan)[0];
                                green = (byte)(dataPtr_o + yo * widthstep + xo * nChan)[1];
                                red = (byte)(dataPtr_o + yo * widthstep + xo * nChan)[2];

                                //store in the copy image
                                dataPtr_c[0] = blue;
                                dataPtr_c[1] = green;
                                dataPtr_c[2] = red;

                            }
                            else
                            {   //store in the copy image
                                dataPtr_c[0] = 0;
                                dataPtr_c[1] = 0;
                                dataPtr_c[2] = 0;
                            }
                            // advance the pointer to the next pixel
                            dataPtr_c += nChan;
                        }
                        //at the end of the line advance the pointer by the aligment bytes (padding)
                        dataPtr_c += padding;
                    }
                }
            }
        }

        /// <summary>
        /// Scale 
        /// </summary>
        /// <param name="img">Image</param>
        public static void Scale(Image<Bgr, byte> imgDestino, Image<Bgr, byte> imgOrigem, float scaleFactor)

        {
            unsafe
            {
                // direct access to the image memory(sequencial)
                // direcion top left -> bottom right

                MIplImage origem = imgOrigem.MIplImage;
                byte* dataPtr_o = (byte*)origem.ImageData.ToPointer(); // Pointer to the image

                MIplImage copy = imgDestino.MIplImage;
                byte* dataPtr_c = (byte*)copy.ImageData.ToPointer(); // Pointer to the image

                int width = origem.Width;
                int height = origem.Height;
                int nChan = origem.NChannels; // number of channels - 3
                int padding = origem.WidthStep - origem.NChannels * origem.Width; // alinhament bytes (padding)
                int x, y, yo, xo;
                int widthstep = origem.WidthStep;
                byte blue, green, red;


                if (nChan == 3) // image in RGB
                {
                    for (y = 0; y < height; y++)
                    {
                        for (x = 0; x < width; x++)
                        {
                            //retrieve 3 color componets from copy
                            xo = (int)(x / scaleFactor);
                            yo = (int)(y / scaleFactor);


                            if ((xo < width && xo >= 0) && (yo < height && yo >= 0))
                            {
                                //retrieve 3 colour components from origem
                                blue = (byte)(dataPtr_o + yo * widthstep + xo * nChan)[0];
                                green = (byte)(dataPtr_o + yo * widthstep + xo * nChan)[1];
                                red = (byte)(dataPtr_o + yo * widthstep + xo * nChan)[2];

                                //store in the copy image
                                dataPtr_c[0] = blue;
                                dataPtr_c[1] = green;
                                dataPtr_c[2] = red;

                            }
                            else
                            {
                                //store in the copy image
                                dataPtr_c[0] = 0;
                                dataPtr_c[1] = 0;
                                dataPtr_c[2] = 0;
                            }
                            // advance the pointer to the next pixel
                            dataPtr_c += nChan;
                        }
                        //at the end of the line advance the pointer by the aligment bytes (padding)
                        dataPtr_c += padding;
                    }
                }
            }
        }

        /// <summary>
        /// Scale centered in point selected by user
        /// </summary>
        /// <param name="img">Image</param>
        public static void Scale_point_xy(Image<Bgr, byte> imgDestino, Image<Bgr, byte> imgOrigem, float scaleFactor, int centerX, int centerY)
        {
            unsafe
            {
                // direct access to the image memory(sequencial)
                // direcion top left -> bottom right

                MIplImage origem = imgOrigem.MIplImage;
                byte* dataPtr_o = (byte*)origem.ImageData.ToPointer(); // Pointer to the image

                MIplImage copy = imgDestino.MIplImage;
                byte* dataPtr_c = (byte*)copy.ImageData.ToPointer(); // Pointer to the image

                int width = origem.Width;
                int height = origem.Height;
                int nChan = origem.NChannels; // number of channels - 3
                int padding = origem.WidthStep - origem.NChannels * origem.Width; // alinhament bytes (padding)
                int x, y, yo, xo;
                int widthstep = origem.WidthStep;
                byte blue, green, red;
                int H = height / 2;
                int W = width / 2;

                if (nChan == 3) // image in RGB
                {
                    for (y = 0; y < height; y++)
                    {
                        for (x = 0; x < width; x++)
                        {
                            //retrieve 3 color componets from copy
                            xo = (int)Math.Round((x / scaleFactor) + centerX - (W / scaleFactor));
                            yo = (int)Math.Round((y / scaleFactor) + centerY - (H / scaleFactor));


                            if ((xo < width && xo >= 0) && (yo < height && yo >= 0))
                            {
                                //retrieve 3 colour components from origem
                                blue = (byte)(dataPtr_o + yo * widthstep + xo * nChan)[0];
                                green = (byte)(dataPtr_o + yo * widthstep + xo * nChan)[1];
                                red = (byte)(dataPtr_o + yo * widthstep + xo * nChan)[2];

                                //store in the copy image
                                dataPtr_c[0] = blue;
                                dataPtr_c[1] = green;
                                dataPtr_c[2] = red;

                            }
                            else
                            {
                                //store in the copy image
                                dataPtr_c[0] = 0;
                                dataPtr_c[1] = 0;
                                dataPtr_c[2] = 0;
                            }
                            // advance the pointer to the next pixel
                            dataPtr_c += nChan;
                        }
                        //at the end of the line advance the pointer by the aligment bytes (padding)
                        dataPtr_c += padding;
                    }
                }
            }
        }

        /// <summary>
        /// Filtro de média 3x3 (método A)
        /// Direct access to memory - faster method
        /// </summary>
        /// <param name="img">Image</param>
        public static void Mean(Image<Bgr, byte> img, Image<Bgr, byte> imgCopy)
        {
            unsafe
            {
                MIplImage origem = img.MIplImage;
                byte* dataPtr_o = (byte*)origem.ImageData.ToPointer(); // Pointer to the image

                MIplImage copy = imgCopy.MIplImage;
                byte* dataPtr_c = (byte*)copy.ImageData.ToPointer(); // Pointer to the image

                int width = origem.Width;
                int height = origem.Height;
                int widthStep = origem.WidthStep;
                int nChan = origem.NChannels; // number of channels = 3
                int x, y, aux_b, aux_g, aux_r, blue, green, red;

                int w = width - 1;
                int h = height - 1;

                /*********** CORE *************/
                for (y = 1; y < h; y++)
                {
                    for (x = 1; x < w; x++)
                    {
                        aux_r = 0;
                        aux_b = 0;
                        aux_g = 0;

                        aux_b += (dataPtr_c + (y - 1) * widthStep + (x - 1) * nChan)[0];
                        aux_b += (dataPtr_c + (y - 1) * widthStep + (x) * nChan)[0];
                        aux_b += (dataPtr_c + (y - 1) * widthStep + (x + 1) * nChan)[0];
                        aux_b += (dataPtr_c + (y) * widthStep + (x - 1) * nChan)[0];
                        aux_b += (dataPtr_c + (y) * widthStep + (x) * nChan)[0];
                        aux_b += (dataPtr_c + (y) * widthStep + (x + 1) * nChan)[0];
                        aux_b += (dataPtr_c + (y + 1) * widthStep + (x - 1) * nChan)[0];
                        aux_b += (dataPtr_c + (y + 1) * widthStep + (x) * nChan)[0];
                        aux_b += (dataPtr_c + (y + 1) * widthStep + (x + 1) * nChan)[0];

                        aux_g += (dataPtr_c + (y - 1) * widthStep + (x - 1) * nChan)[1];
                        aux_g += (dataPtr_c + (y - 1) * widthStep + (x) * nChan)[1];
                        aux_g += (dataPtr_c + (y - 1) * widthStep + (x + 1) * nChan)[1];
                        aux_g += (dataPtr_c + (y) * widthStep + (x - 1) * nChan)[1];
                        aux_g += (dataPtr_c + (y) * widthStep + (x) * nChan)[1];
                        aux_g += (dataPtr_c + (y) * widthStep + (x + 1) * nChan)[1];
                        aux_g += (dataPtr_c + (y + 1) * widthStep + (x - 1) * nChan)[1];
                        aux_g += (dataPtr_c + (y + 1) * widthStep + (x) * nChan)[1];
                        aux_g += (dataPtr_c + (y + 1) * widthStep + (x + 1) * nChan)[1];

                        aux_r += (dataPtr_c + (y - 1) * widthStep + (x - 1) * nChan)[2];
                        aux_r += (dataPtr_c + (y - 1) * widthStep + (x) * nChan)[2];
                        aux_r += (dataPtr_c + (y - 1) * widthStep + (x + 1) * nChan)[2];
                        aux_r += (dataPtr_c + (y) * widthStep + (x - 1) * nChan)[2];
                        aux_r += (dataPtr_c + (y) * widthStep + (x) * nChan)[2];
                        aux_r += (dataPtr_c + (y) * widthStep + (x + 1) * nChan)[2];
                        aux_r += (dataPtr_c + (y + 1) * widthStep + (x - 1) * nChan)[2];
                        aux_r += (dataPtr_c + (y + 1) * widthStep + (x) * nChan)[2];
                        aux_r += (dataPtr_c + (y + 1) * widthStep + (x + 1) * nChan)[2];

                        blue = (int)Math.Round(aux_b / 9.0);
                        green = (int)Math.Round(aux_g / 9.0);
                        red = (int)Math.Round(aux_r / 9.0);

                        (dataPtr_o + (y * widthStep) + (x * nChan))[0] = (byte)blue;
                        (dataPtr_o + (y * widthStep) + (x * nChan))[1] = (byte)green;
                        (dataPtr_o + (y * widthStep) + (x * nChan))[2] = (byte)red;
                    }
                }

                /**************************** INFERIOR ******************************/

                for (x = 1; x < w; x++)
                {
                    y = h;
                    aux_r = 0;
                    aux_b = 0;
                    aux_g = 0;

                    aux_b += 2 * (dataPtr_c + (y) * widthStep + (x - 1) * nChan)[0];
                    aux_b += 2 * (dataPtr_c + (y) * widthStep + (x) * nChan)[0];
                    aux_b += 2 * (dataPtr_c + (y) * widthStep + (x + 1) * nChan)[0];

                    aux_b += (dataPtr_c + (h - 1) * widthStep + (x - 1) * nChan)[0];
                    aux_b += (dataPtr_c + (h - 1) * widthStep + (x) * nChan)[0];
                    aux_b += (dataPtr_c + (h - 1) * widthStep + (x + 1) * nChan)[0];


                    aux_g += 2 * (dataPtr_c + (y) * widthStep + (x - 1) * nChan)[1];
                    aux_g += 2 * (dataPtr_c + (y) * widthStep + (x) * nChan)[1];
                    aux_g += 2 * (dataPtr_c + (y) * widthStep + (x + 1) * nChan)[1];

                    aux_g += (dataPtr_c + (h - 1) * widthStep + (x - 1) * nChan)[1];
                    aux_g += (dataPtr_c + (h - 1) * widthStep + (x) * nChan)[1];
                    aux_g += (dataPtr_c + (h - 1) * widthStep + (x + 1) * nChan)[1];


                    aux_r += 2 * (dataPtr_c + (y) * widthStep + (x - 1) * nChan)[2];
                    aux_r += 2 * (dataPtr_c + (y) * widthStep + (x) * nChan)[2];
                    aux_r += 2 * (dataPtr_c + (y) * widthStep + (x + 1) * nChan)[2];

                    aux_r += (dataPtr_c + (h - 1) * widthStep + (x - 1) * nChan)[2];
                    aux_r += (dataPtr_c + (h - 1) * widthStep + (x) * nChan)[2];
                    aux_r += (dataPtr_c + (h - 1) * widthStep + (x + 1) * nChan)[2];

                    blue = (int)Math.Round(aux_b / 9.0);
                    green = (int)Math.Round(aux_g / 9.0);
                    red = (int)Math.Round(aux_r / 9.0);

                    (dataPtr_o + (y * widthStep) + (x * nChan))[0] = (byte)blue;
                    (dataPtr_o + (y * widthStep) + (x * nChan))[1] = (byte)green;
                    (dataPtr_o + (y * widthStep) + (x * nChan))[2] = (byte)red;
                }

                /**************************** SUPERIOR ******************************/

                for (x = 1; x < w; x++)
                {
                    y = 0;
                    aux_r = 0;
                    aux_b = 0;
                    aux_g = 0;

                    aux_b += 2 * (dataPtr_c + (x - 1) * nChan)[0];
                    aux_b += 2 * (dataPtr_c + (x) * nChan)[0];
                    aux_b += 2 * (dataPtr_c + (x + 1) * nChan)[0];

                    aux_b += (dataPtr_c + (1) * widthStep + (x - 1) * nChan)[0];
                    aux_b += (dataPtr_c + (1) * widthStep + (x) * nChan)[0];
                    aux_b += (dataPtr_c + (1) * widthStep + (x + 1) * nChan)[0];


                    aux_g += 2 * (dataPtr_c + (x - 1) * nChan)[1];
                    aux_g += 2 * (dataPtr_c + (x) * nChan)[1];
                    aux_g += 2 * (dataPtr_c + (x + 1) * nChan)[1];

                    aux_g += (dataPtr_c + (1) * widthStep + (x - 1) * nChan)[1];
                    aux_g += (dataPtr_c + (1) * widthStep + (x) * nChan)[1];
                    aux_g += (dataPtr_c + (1) * widthStep + (x + 1) * nChan)[1];


                    aux_r += 2 * (dataPtr_c + (x - 1) * nChan)[2];
                    aux_r += 2 * (dataPtr_c + (x) * nChan)[2];
                    aux_r += 2 * (dataPtr_c + (x + 1) * nChan)[2];

                    aux_r += (dataPtr_c + (1) * widthStep + (x - 1) * nChan)[2];
                    aux_r += (dataPtr_c + (1) * widthStep + (x) * nChan)[2];
                    aux_r += (dataPtr_c + (1) * widthStep + (x + 1) * nChan)[2];

                    blue = (int)Math.Round(aux_b / 9.0);
                    green = (int)Math.Round(aux_g / 9.0);
                    red = (int)Math.Round(aux_r / 9.0);

                    (dataPtr_o + (y * widthStep) + (x * nChan))[0] = (byte)blue;
                    (dataPtr_o + (y * widthStep) + (x * nChan))[1] = (byte)green;
                    (dataPtr_o + (y * widthStep) + (x * nChan))[2] = (byte)red;

                }
                /**************************** ESQUERDA ******************************/

                for (y = 1; y < h; y++)
                {
                    x = 0;
                    aux_r = 0;
                    aux_b = 0;
                    aux_g = 0;

                    aux_b += 2 * (dataPtr_c + (y - 1) * widthStep)[0];
                    aux_b += 2 * (dataPtr_c + (y) * widthStep)[0];
                    aux_b += 2 * (dataPtr_c + (y + 1) * widthStep)[0];

                    aux_b += (dataPtr_c + (y - 1) * widthStep + nChan)[0];
                    aux_b += (dataPtr_c + (y) * widthStep + nChan)[0];
                    aux_b += (dataPtr_c + (y + 1) * widthStep + nChan)[0];


                    aux_g += 2 * (dataPtr_c + (y - 1) * widthStep)[1];
                    aux_g += 2 * (dataPtr_c + (y) * widthStep)[1];
                    aux_g += 2 * (dataPtr_c + (y + 1) * widthStep)[1];

                    aux_g += (dataPtr_c + (y - 1) * widthStep + nChan)[1];
                    aux_g += (dataPtr_c + (y) * widthStep + nChan)[1];
                    aux_g += (dataPtr_c + (y + 1) * widthStep + nChan)[1];


                    aux_r += 2 * (dataPtr_c + (y - 1) * widthStep)[2];
                    aux_r += 2 * (dataPtr_c + (y) * widthStep)[2];
                    aux_r += 2 * (dataPtr_c + (y + 1) * widthStep)[2];

                    aux_r += (dataPtr_c + (y - 1) * widthStep + nChan)[2];
                    aux_r += (dataPtr_c + (y) * widthStep + nChan)[2];
                    aux_r += (dataPtr_c + (y + 1) * widthStep + nChan)[2];

                    blue = (int)Math.Round(aux_b / 9.0);
                    green = (int)Math.Round(aux_g / 9.0);
                    red = (int)Math.Round(aux_r / 9.0);

                    (dataPtr_o + (y * widthStep) + (x * nChan))[0] = (byte)blue;
                    (dataPtr_o + (y * widthStep) + (x * nChan))[1] = (byte)green;
                    (dataPtr_o + (y * widthStep) + (x * nChan))[2] = (byte)red;
                }

                /**************************** DIREITO ******************************/

                for (y = 1; y < h; y++)
                {
                    x = w;
                    aux_r = 0;
                    aux_b = 0;
                    aux_g = 0;

                    aux_b += 2 * (dataPtr_c + (y - 1) * widthStep + (x) * nChan)[0];
                    aux_b += 2 * (dataPtr_c + (y) * widthStep + (x) * nChan)[0];
                    aux_b += 2 * (dataPtr_c + (y + 1) * widthStep + (x) * nChan)[0];

                    aux_b += (dataPtr_c + (y - 1) * widthStep + (x - 1) * nChan)[0];
                    aux_b += (dataPtr_c + (y) * widthStep + (x - 1) * nChan)[0];
                    aux_b += (dataPtr_c + (y + 1) * widthStep + (x - 1) * nChan)[0];


                    aux_g += 2 * (dataPtr_c + (y - 1) * widthStep + (x) * nChan)[1];
                    aux_g += 2 * (dataPtr_c + (y) * widthStep + (x) * nChan)[1];
                    aux_g += 2 * (dataPtr_c + (y + 1) * widthStep + (x) * nChan)[1];

                    aux_g += (dataPtr_c + (y - 1) * widthStep + (x - 1) * nChan)[1];
                    aux_g += (dataPtr_c + (y) * widthStep + (x - 1) * nChan)[1];
                    aux_g += (dataPtr_c + (y + 1) * widthStep + (x - 1) * nChan)[1];


                    aux_r += 2 * (dataPtr_c + (y - 1) * widthStep + (x) * nChan)[2];
                    aux_r += 2 * (dataPtr_c + (y) * widthStep + (x) * nChan)[2];
                    aux_r += 2 * (dataPtr_c + (y + 1) * widthStep + (x) * nChan)[2];

                    aux_r += (dataPtr_c + (y - 1) * widthStep + (x - 1) * nChan)[2];
                    aux_r += (dataPtr_c + (y) * widthStep + (x - 1) * nChan)[2];
                    aux_r += (dataPtr_c + (y + 1) * widthStep + (x - 1) * nChan)[2];

                    blue = (int)Math.Round(aux_b / 9.0);
                    green = (int)Math.Round(aux_g / 9.0);
                    red = (int)Math.Round(aux_r / 9.0);

                    (dataPtr_o + (y * widthStep) + (x * nChan))[0] = (byte)blue;
                    (dataPtr_o + (y * widthStep) + (x * nChan))[1] = (byte)green;
                    (dataPtr_o + (y * widthStep) + (x * nChan))[2] = (byte)red;
                }

                /************************* CANTO SUPERIOR ESQUERDO ****************************/

                aux_r = 0;
                aux_b = 0;
                aux_g = 0;

                aux_b += 4 * (dataPtr_c)[0];
                aux_b += 2 * (dataPtr_c + widthStep)[0];
                aux_b += 2 * (dataPtr_c + nChan)[0];
                aux_b += (dataPtr_c + widthStep + nChan)[0];


                aux_g += 4 * (dataPtr_c)[1];
                aux_g += 2 * (dataPtr_c + widthStep)[1];
                aux_g += 2 * (dataPtr_c + nChan)[1];
                aux_g += (dataPtr_c + widthStep + nChan)[1];


                aux_r += 4 * (dataPtr_c)[2];
                aux_r += 2 * (dataPtr_c + widthStep)[2];
                aux_r += 2 * (dataPtr_c + nChan)[2];
                aux_r += (dataPtr_c + widthStep + nChan)[2];


                blue = (int)Math.Round(aux_b / 9.0);
                green = (int)Math.Round(aux_g / 9.0);
                red = (int)Math.Round(aux_r / 9.0);

                (dataPtr_o)[0] = (byte)blue;
                (dataPtr_o)[1] = (byte)green;
                (dataPtr_o)[2] = (byte)red;

                /************************* CANTO INFERIOR ESQUERDO ****************************/

                aux_r = 0;
                aux_b = 0;
                aux_g = 0;

                aux_b += 4 * (dataPtr_c + h * widthStep)[0];
                aux_b += 2 * (dataPtr_c + (h - 1) * widthStep)[0];
                aux_b += 2 * (dataPtr_c + h * widthStep + nChan)[0];
                aux_b += (dataPtr_c + (h - 1) * widthStep + nChan)[0];


                aux_g += 4 * (dataPtr_c + h * widthStep)[1];
                aux_g += 2 * (dataPtr_c + (h - 1) * widthStep)[1];
                aux_g += 2 * (dataPtr_c + h * widthStep + nChan)[1];
                aux_g += (dataPtr_c + (h - 1) * widthStep + nChan)[1];


                aux_r += 4 * (dataPtr_c + h * widthStep)[2];
                aux_r += 2 * (dataPtr_c + (h - 1) * widthStep)[2];
                aux_r += 2 * (dataPtr_c + h * widthStep + nChan)[2];
                aux_r += (dataPtr_c + (h - 1) * widthStep + nChan)[2];


                blue = (int)Math.Round(aux_b / 9.0);
                green = (int)Math.Round(aux_g / 9.0);
                red = (int)Math.Round(aux_r / 9.0);

                (dataPtr_o + (h * widthStep))[0] = (byte)blue;
                (dataPtr_o + (h * widthStep))[1] = (byte)green;
                (dataPtr_o + (h * widthStep))[2] = (byte)red;

                /*************************CANTO SUPERIOR DIREITO ****************************/

                aux_r = 0;
                aux_b = 0;
                aux_g = 0;

                aux_b += 4 * (dataPtr_c + w * nChan)[0];
                aux_b += 2 * (dataPtr_c + (w - 1) * nChan)[0];
                aux_b += 2 * (dataPtr_c + widthStep + w * nChan)[0];
                aux_b += (dataPtr_c + widthStep + (w - 1) * nChan)[0];


                aux_g += 4 * (dataPtr_c + w * nChan)[1];
                aux_g += 2 * (dataPtr_c + (w - 1) * nChan)[1];
                aux_g += 2 * (dataPtr_c + widthStep + w * nChan)[1];
                aux_g += (dataPtr_c + widthStep + (w - 1) * nChan)[1];


                aux_r += 4 * (dataPtr_c + w * nChan)[2];
                aux_r += 2 * (dataPtr_c + (w - 1) * nChan)[2];
                aux_r += 2 * (dataPtr_c + widthStep + w * nChan)[2];
                aux_r += (dataPtr_c + widthStep + (w - 1) * nChan)[2];


                blue = (int)Math.Round(aux_b / 9.0);
                green = (int)Math.Round(aux_g / 9.0);
                red = (int)Math.Round(aux_r / 9.0);

                (dataPtr_o + (w * nChan))[0] = (byte)blue;
                (dataPtr_o + (w * nChan))[1] = (byte)green;
                (dataPtr_o + (w * nChan))[2] = (byte)red;

                /*************************CANTO INFERIOR DIREITO ****************************/

                aux_r = 0;
                aux_b = 0;
                aux_g = 0;

                aux_b += 4 * (dataPtr_c + h * widthStep + w * nChan)[0];
                aux_b += 2 * (dataPtr_c + h * widthStep + (w - 1) * nChan)[0];
                aux_b += 2 * (dataPtr_c + (h - 1) * widthStep + w * nChan)[0];
                aux_b += (dataPtr_c + (h - 1) * widthStep + (w - 1) * nChan)[0];


                aux_g += 4 * (dataPtr_c + h * widthStep + w * nChan)[1];
                aux_g += 2 * (dataPtr_c + h * widthStep + (w - 1) * nChan)[1];
                aux_g += 2 * (dataPtr_c + (h - 1) * widthStep + w * nChan)[1];
                aux_g += (dataPtr_c + (h - 1) * widthStep + (w - 1) * nChan)[1];


                aux_r += 4 * (dataPtr_c + h * widthStep + w * nChan)[2];
                aux_r += 2 * (dataPtr_c + h * widthStep + (w - 1) * nChan)[2];
                aux_r += 2 * (dataPtr_c + (h - 1) * widthStep + w * nChan)[2];
                aux_r += (dataPtr_c + (h - 1) * widthStep + (w - 1) * nChan)[2];


                blue = (int)Math.Round(aux_b / 9.0);
                green = (int)Math.Round(aux_g / 9.0);
                red = (int)Math.Round(aux_r / 9.0);

                (dataPtr_o + (h * widthStep) + (w * nChan))[0] = (byte)blue;
                (dataPtr_o + (h * widthStep) + (w * nChan))[1] = (byte)green;
                (dataPtr_o + (h * widthStep) + (w * nChan))[2] = (byte)red;
            }
        }

        /// <summary>
        /// Filtro de média não uniforme 3x3 (método A)
        /// Direct access to memory - faster method
        /// </summary>
        /// <param name="img">Image</param>
        public static void NonUniform(Image<Bgr, byte> img, Image<Bgr, byte> imgCopy, float[,] matrix, float matrixWeight, float offset)
        {
            unsafe
            {
                MIplImage m = img.MIplImage;
                byte* dataPtr = (byte*)m.ImageData.ToPointer();
                MIplImage mCopy = imgCopy.MIplImage;
                byte* copyPtr = (byte*)mCopy.ImageData.ToPointer();

                int x, y;
                int height = img.Height;
                int width = img.Width;
                int nChan = m.NChannels; // number of channels - 3
                int padding = m.WidthStep - m.Width * m.NChannels;
                int widthstep = m.WidthStep;
                byte* pixel;
                double sumBlue, sumGreen, sumRed;


                if (nChan == 3)
                {

                    for (y = 1; y < height - 1; y++)
                    {

                        for (x = 1; x < width - 1; x++)
                        {

                            sumBlue = matrix[0, 0] * (copyPtr + (y - 1) * widthstep + (x - 1) * nChan)[0]
                                + matrix[0, 1] * (copyPtr + (y - 1) * widthstep + x * nChan)[0]
                                + matrix[0, 2] * (copyPtr + (y - 1) * widthstep + (x + 1) * nChan)[0]
                                + matrix[1, 0] * (copyPtr + y * widthstep + (x - 1) * nChan)[0]
                                + matrix[1, 1] * (copyPtr + y * widthstep + x * nChan)[0]
                                + matrix[1, 2] * (copyPtr + y * widthstep + (x + 1) * nChan)[0]
                                + matrix[2, 0] * (copyPtr + (y + 1) * widthstep + (x - 1) * nChan)[0]
                                + matrix[2, 1] * (copyPtr + (y + 1) * widthstep + x * nChan)[0]
                                + matrix[2, 2] * (copyPtr + (y + 1) * widthstep + (x + 1) * nChan)[0];

                            sumGreen = matrix[0, 0] * (copyPtr + (y - 1) * widthstep + (x - 1) * nChan)[1]
                                + matrix[0, 1] * (copyPtr + (y - 1) * widthstep + x * nChan)[1]
                                + matrix[0, 2] * (copyPtr + (y - 1) * widthstep + (x + 1) * nChan)[1]
                                + matrix[1, 0] * (copyPtr + y * widthstep + (x - 1) * nChan)[1]
                                + matrix[1, 1] * (copyPtr + y * widthstep + x * nChan)[1]
                                + matrix[1, 2] * (copyPtr + y * widthstep + (x + 1) * nChan)[1]
                                + matrix[2, 0] * (copyPtr + (y + 1) * widthstep + (x - 1) * nChan)[1]
                                + matrix[2, 1] * (copyPtr + (y + 1) * widthstep + x * nChan)[1]
                                + matrix[2, 2] * (copyPtr + (y + 1) * widthstep + (x + 1) * nChan)[1];

                            sumRed = matrix[0, 0] * (copyPtr + (y - 1) * widthstep + (x - 1) * nChan)[2]
                                + matrix[0, 1] * (copyPtr + (y - 1) * widthstep + x * nChan)[2]
                                + matrix[0, 2] * (copyPtr + (y - 1) * widthstep + (x + 1) * nChan)[2]
                                + matrix[1, 0] * (copyPtr + y * widthstep + (x - 1) * nChan)[2]
                                + matrix[1, 1] * (copyPtr + y * widthstep + x * nChan)[2]
                                + matrix[1, 2] * (copyPtr + y * widthstep + (x + 1) * nChan)[2]
                                + matrix[2, 0] * (copyPtr + (y + 1) * widthstep + (x - 1) * nChan)[2]
                                + matrix[2, 1] * (copyPtr + (y + 1) * widthstep + x * nChan)[2]
                                + matrix[2, 2] * (copyPtr + (y + 1) * widthstep + (x + 1) * nChan)[2];


                            sumBlue = Math.Round((sumBlue / matrixWeight) + offset);
                            sumGreen = Math.Round((sumGreen / matrixWeight) + offset);
                            sumRed = Math.Round((sumRed / matrixWeight) + offset);

                            if (sumBlue < 0)
                                sumBlue = 0;
                            if (sumBlue > 255)
                                sumBlue = 255;

                            if (sumGreen < 0)
                                sumGreen = 0;
                            if (sumGreen > 255)
                                sumGreen = 255;

                            if (sumRed < 0)
                                sumRed = 0;
                            if (sumRed > 255)
                                sumRed = 255;

                            pixel = dataPtr + y * widthstep + x * nChan;
                            pixel[0] = (byte)sumBlue;
                            pixel[1] = (byte)sumGreen;
                            pixel[2] = (byte)sumRed;
                        }

                    }

                    // primeira linha, sem os cantos

                    for (x = 1; x < width - 1; x++)
                    {
                        sumBlue = matrix[0, 0] * (copyPtr + (x - 1) * nChan)[0]
                            + matrix[0, 1] * (copyPtr + x * nChan)[0]
                            + matrix[0, 2] * (copyPtr + (x + 1) * nChan)[0]
                            + matrix[1, 0] * (copyPtr + (x - 1) * nChan)[0]
                            + matrix[1, 1] * (copyPtr + x * nChan)[0]
                            + matrix[1, 2] * (copyPtr + (x + 1) * nChan)[0]
                            + matrix[2, 0] * (copyPtr + widthstep + (x - 1) * nChan)[0]
                            + matrix[2, 1] * (copyPtr + widthstep + x * nChan)[0]
                            + matrix[2, 2] * (copyPtr + widthstep + (x + 1) * nChan)[0];

                        sumGreen = matrix[0, 0] * (copyPtr + (x - 1) * nChan)[1]
                            + matrix[0, 1] * (copyPtr + x * nChan)[1]
                            + matrix[0, 2] * (copyPtr + (x + 1) * nChan)[1]
                            + matrix[1, 0] * (copyPtr + (x - 1) * nChan)[1]
                            + matrix[1, 1] * (copyPtr + x * nChan)[1]
                            + matrix[1, 2] * (copyPtr + (x + 1) * nChan)[1]
                            + matrix[2, 0] * (copyPtr + widthstep + (x - 1) * nChan)[1]
                            + matrix[2, 1] * (copyPtr + widthstep + x * nChan)[1]
                            + matrix[2, 2] * (copyPtr + widthstep + (x + 1) * nChan)[1];

                        sumRed = matrix[0, 0] * (copyPtr + (x - 1) * nChan)[2]
                            + matrix[0, 1] * (copyPtr + x * nChan)[2]
                            + matrix[0, 2] * (copyPtr + (x + 1) * nChan)[2]
                            + matrix[1, 0] * (copyPtr + (x - 1) * nChan)[2]
                            + matrix[1, 1] * (copyPtr + x * nChan)[2]
                            + matrix[1, 2] * (copyPtr + (x + 1) * nChan)[2]
                            + matrix[2, 0] * (copyPtr + widthstep + (x - 1) * nChan)[2]
                            + matrix[2, 1] * (copyPtr + widthstep + x * nChan)[2]
                            + matrix[2, 2] * (copyPtr + widthstep + (x + 1) * nChan)[2];

                        sumBlue = Math.Round((sumBlue / matrixWeight) + offset);
                        sumGreen = Math.Round((sumGreen / matrixWeight) + offset);
                        sumRed = Math.Round((sumRed / matrixWeight) + offset);

                        if (sumBlue < 0)
                            sumBlue = 0;
                        if (sumBlue > 255)
                            sumBlue = 255;

                        if (sumGreen < 0)
                            sumGreen = 0;
                        if (sumGreen > 255)
                            sumGreen = 255;

                        if (sumRed < 0)
                            sumRed = 0;
                        if (sumRed > 255)
                            sumRed = 255;

                        pixel = dataPtr + x * nChan;
                        pixel[0] = (byte)sumBlue;
                        pixel[1] = (byte)sumGreen;
                        pixel[2] = (byte)sumRed;

                    }


                    // ultima linha, sem os cantos

                    for (x = 1; x < width - 1; x++)
                    {
                        sumBlue = matrix[0, 0] * (copyPtr + (height - 2) * widthstep + (x - 1) * nChan)[0]
                            + matrix[0, 1] * (copyPtr + (height - 2) * widthstep + x * nChan)[0]
                            + matrix[0, 2] * (copyPtr + (height - 2) * widthstep + (x + 1) * nChan)[0]
                            + matrix[1, 0] * (copyPtr + (height - 1) * widthstep + (x - 1) * nChan)[0]
                            + matrix[1, 1] * (copyPtr + (height - 1) * widthstep + x * nChan)[0]
                            + matrix[1, 2] * (copyPtr + (height - 1) * widthstep + (x + 1) * nChan)[0]
                            + matrix[2, 0] * (copyPtr + (height - 1) * widthstep + (x - 1) * nChan)[0]
                            + matrix[2, 1] * (copyPtr + (height - 1) * widthstep + x * nChan)[0]
                            + matrix[2, 2] * (copyPtr + (height - 1) * widthstep + (x + 1) * nChan)[0];

                        sumGreen = matrix[0, 0] * (copyPtr + (height - 2) * widthstep + (x - 1) * nChan)[1]
                            + matrix[0, 1] * (copyPtr + (height - 2) * widthstep + x * nChan)[1]
                            + matrix[0, 2] * (copyPtr + (height - 2) * widthstep + (x + 1) * nChan)[1]
                            + matrix[1, 0] * (copyPtr + (height - 1) * widthstep + (x - 1) * nChan)[1]
                            + matrix[1, 1] * (copyPtr + (height - 1) * widthstep + x * nChan)[1]
                            + matrix[1, 2] * (copyPtr + (height - 1) * widthstep + (x + 1) * nChan)[1]
                            + matrix[2, 0] * (copyPtr + (height - 1) * widthstep + (x - 1) * nChan)[1]
                            + matrix[2, 1] * (copyPtr + (height - 1) * widthstep + x * nChan)[1]
                            + matrix[2, 2] * (copyPtr + (height - 1) * widthstep + (x + 1) * nChan)[1];

                        sumRed = matrix[0, 0] * (copyPtr + (height - 2) * widthstep + (x - 1) * nChan)[2]
                            + matrix[0, 1] * (copyPtr + (height - 2) * widthstep + x * nChan)[2]
                            + matrix[0, 2] * (copyPtr + (height - 2) * widthstep + (x + 1) * nChan)[2]
                            + matrix[1, 0] * (copyPtr + (height - 1) * widthstep + (x - 1) * nChan)[2]
                            + matrix[1, 1] * (copyPtr + (height - 1) * widthstep + x * nChan)[2]
                            + matrix[1, 2] * (copyPtr + (height - 1) * widthstep + (x + 1) * nChan)[2]
                            + matrix[2, 0] * (copyPtr + (height - 1) * widthstep + (x - 1) * nChan)[2]
                            + matrix[2, 1] * (copyPtr + (height - 1) * widthstep + x * nChan)[2]
                            + matrix[2, 2] * (copyPtr + (height - 1) * widthstep + (x + 1) * nChan)[2];

                        sumBlue = Math.Round((sumBlue / matrixWeight) + offset);
                        sumGreen = Math.Round((sumGreen / matrixWeight) + offset);
                        sumRed = Math.Round((sumRed / matrixWeight) + offset);

                        if (sumBlue < 0)
                            sumBlue = 0;
                        if (sumBlue > 255)
                            sumBlue = 255;

                        if (sumGreen < 0)
                            sumGreen = 0;
                        if (sumGreen > 255)
                            sumGreen = 255;

                        if (sumRed < 0)
                            sumRed = 0;
                        if (sumRed > 255)
                            sumRed = 255;

                        pixel = dataPtr + (height - 1) * widthstep + x * nChan;
                        pixel[0] = (byte)sumBlue;
                        pixel[1] = (byte)sumGreen;
                        pixel[2] = (byte)sumRed;

                    }


                    // primeira coluna, sem os cantos

                    for (y = 1; y < height - 1; y++)
                    {
                        sumBlue = matrix[0, 0] * (copyPtr + (y - 1) * widthstep)[0]
                            + matrix[0, 1] * (copyPtr + (y - 1) * widthstep)[0]
                            + matrix[0, 2] * (copyPtr + (y - 1) * widthstep + nChan)[0]
                            + matrix[1, 0] * (copyPtr + y * widthstep)[0]
                            + matrix[1, 1] * (copyPtr + y * widthstep)[0]
                            + matrix[1, 2] * (copyPtr + y * widthstep + nChan)[0]
                            + matrix[2, 0] * (copyPtr + (y + 1) * widthstep)[0]
                            + matrix[2, 1] * (copyPtr + (y + 1) * widthstep)[0]
                            + matrix[2, 2] * (copyPtr + (y + 1) * widthstep + nChan)[0];

                        sumGreen = matrix[0, 0] * (copyPtr + (y - 1) * widthstep)[1]
                            + matrix[0, 1] * (copyPtr + (y - 1) * widthstep)[1]
                            + matrix[0, 2] * (copyPtr + (y - 1) * widthstep + nChan)[1]
                            + matrix[1, 0] * (copyPtr + y * widthstep)[1]
                            + matrix[1, 1] * (copyPtr + y * widthstep)[1]
                            + matrix[1, 2] * (copyPtr + y * widthstep + nChan)[1]
                            + matrix[2, 0] * (copyPtr + (y + 1) * widthstep)[1]
                            + matrix[2, 1] * (copyPtr + (y + 1) * widthstep)[1]
                            + matrix[2, 2] * (copyPtr + (y + 1) * widthstep + nChan)[1];

                        sumRed = matrix[0, 0] * (copyPtr + (y - 1) * widthstep)[2]
                            + matrix[0, 1] * (copyPtr + (y - 1) * widthstep)[2]
                            + matrix[0, 2] * (copyPtr + (y - 1) * widthstep + nChan)[2]
                            + matrix[1, 0] * (copyPtr + y * widthstep)[2]
                            + matrix[1, 1] * (copyPtr + y * widthstep)[2]
                            + matrix[1, 2] * (copyPtr + y * widthstep + nChan)[2]
                            + matrix[2, 0] * (copyPtr + (y + 1) * widthstep)[2]
                            + matrix[2, 1] * (copyPtr + (y + 1) * widthstep)[2]
                            + matrix[2, 2] * (copyPtr + (y + 1) * widthstep + nChan)[2];

                        sumBlue = Math.Round((sumBlue / matrixWeight) + offset);
                        sumGreen = Math.Round((sumGreen / matrixWeight) + offset);
                        sumRed = Math.Round((sumRed / matrixWeight) + offset);

                        if (sumBlue < 0)
                            sumBlue = 0;
                        if (sumBlue > 255)
                            sumBlue = 255;

                        if (sumGreen < 0)
                            sumGreen = 0;
                        if (sumGreen > 255)
                            sumGreen = 255;

                        if (sumRed < 0)
                            sumRed = 0;
                        if (sumRed > 255)
                            sumRed = 255;

                        pixel = dataPtr + y * widthstep;
                        pixel[0] = (byte)sumBlue;
                        pixel[1] = (byte)sumGreen;
                        pixel[2] = (byte)sumRed;

                    }


                    // ultima coluna, sem os cantos

                    for (y = 1; y < height - 1; y++)
                    {
                        sumBlue = matrix[0, 0] * (copyPtr + (y - 1) * widthstep + (width - 2) * nChan)[0]
                            + matrix[0, 1] * (copyPtr + (y - 1) * widthstep + (width - 1) * nChan)[0]
                            + matrix[0, 2] * (copyPtr + (y - 1) * widthstep + (width - 1) * nChan)[0]
                            + matrix[1, 0] * (copyPtr + y * widthstep + (width - 2) * nChan)[0]
                            + matrix[1, 1] * (copyPtr + y * widthstep + (width - 1) * nChan)[0]
                            + matrix[1, 2] * (copyPtr + y * widthstep + (width - 1) * nChan)[0]
                            + matrix[2, 0] * (copyPtr + (y + 1) * widthstep + (width - 2) * nChan)[0]
                            + matrix[2, 1] * (copyPtr + (y + 1) * widthstep + (width - 1) * nChan)[0]
                            + matrix[2, 2] * (copyPtr + (y + 1) * widthstep + (width - 1) * nChan)[0];

                        sumGreen = matrix[0, 0] * (copyPtr + (y - 1) * widthstep + (width - 2) * nChan)[1]
                            + matrix[0, 1] * (copyPtr + (y - 1) * widthstep + (width - 1) * nChan)[1]
                            + matrix[0, 2] * (copyPtr + (y - 1) * widthstep + (width - 1) * nChan)[1]
                            + matrix[1, 0] * (copyPtr + y * widthstep + (width - 2) * nChan)[1]
                            + matrix[1, 1] * (copyPtr + y * widthstep + (width - 1) * nChan)[1]
                            + matrix[1, 2] * (copyPtr + y * widthstep + (width - 1) * nChan)[1]
                            + matrix[2, 0] * (copyPtr + (y + 1) * widthstep + (width - 2) * nChan)[1]
                            + matrix[2, 1] * (copyPtr + (y + 1) * widthstep + (width - 1) * nChan)[1]
                            + matrix[2, 2] * (copyPtr + (y + 1) * widthstep + (width - 1) * nChan)[1];

                        sumRed = matrix[0, 0] * (copyPtr + (y - 1) * widthstep + (width - 2) * nChan)[2]
                            + matrix[0, 1] * (copyPtr + (y - 1) * widthstep + (width - 1) * nChan)[2]
                            + matrix[0, 2] * (copyPtr + (y - 1) * widthstep + (width - 1) * nChan)[2]
                            + matrix[1, 0] * (copyPtr + y * widthstep + (width - 2) * nChan)[2]
                            + matrix[1, 1] * (copyPtr + y * widthstep + (width - 1) * nChan)[2]
                            + matrix[1, 2] * (copyPtr + y * widthstep + (width - 1) * nChan)[2]
                            + matrix[2, 0] * (copyPtr + (y + 1) * widthstep + (width - 2) * nChan)[2]
                            + matrix[2, 1] * (copyPtr + (y + 1) * widthstep + (width - 1) * nChan)[2]
                            + matrix[2, 2] * (copyPtr + (y + 1) * widthstep + (width - 1) * nChan)[2];

                        sumBlue = Math.Round((sumBlue / matrixWeight) + offset);
                        sumGreen = Math.Round((sumGreen / matrixWeight) + offset);
                        sumRed = Math.Round((sumRed / matrixWeight) + offset);

                        if (sumBlue < 0)
                            sumBlue = 0;
                        if (sumBlue > 255)
                            sumBlue = 255;

                        if (sumGreen < 0)
                            sumGreen = 0;
                        if (sumGreen > 255)
                            sumGreen = 255;

                        if (sumRed < 0)
                            sumRed = 0;
                        if (sumRed > 255)
                            sumRed = 255;

                        pixel = dataPtr + y * widthstep + (width - 1) * nChan;
                        pixel[0] = (byte)sumBlue;
                        pixel[1] = (byte)sumGreen;
                        pixel[2] = (byte)sumRed;

                    }


                    // canto (0,0)
                    sumBlue = matrix[0, 0] * (copyPtr)[0]
                            + matrix[0, 1] * (copyPtr)[0]
                            + matrix[0, 2] * (copyPtr + nChan)[0]
                            + matrix[1, 0] * (copyPtr)[0]
                            + matrix[1, 1] * (copyPtr)[0]
                            + matrix[1, 2] * (copyPtr + nChan)[0]
                            + matrix[2, 0] * (copyPtr + widthstep)[0]
                            + matrix[2, 1] * (copyPtr + widthstep)[0]
                            + matrix[2, 2] * (copyPtr + widthstep + nChan)[0];

                    sumGreen = matrix[0, 0] * (copyPtr)[1]
                            + matrix[0, 1] * (copyPtr)[1]
                            + matrix[0, 2] * (copyPtr + nChan)[1]
                            + matrix[1, 0] * (copyPtr)[1]
                            + matrix[1, 1] * (copyPtr)[1]
                            + matrix[1, 2] * (copyPtr + nChan)[1]
                            + matrix[2, 0] * (copyPtr + widthstep)[1]
                            + matrix[2, 1] * (copyPtr + widthstep)[1]
                            + matrix[2, 2] * (copyPtr + widthstep + nChan)[1];

                    sumRed = matrix[0, 0] * (copyPtr)[2]
                            + matrix[0, 1] * (copyPtr)[2]
                            + matrix[0, 2] * (copyPtr + nChan)[2]
                            + matrix[1, 0] * (copyPtr)[2]
                            + matrix[1, 1] * (copyPtr)[2]
                            + matrix[1, 2] * (copyPtr + nChan)[2]
                            + matrix[2, 0] * (copyPtr + widthstep)[2]
                            + matrix[2, 1] * (copyPtr + widthstep)[2]
                            + matrix[2, 2] * (copyPtr + widthstep + nChan)[2];

                    sumBlue = Math.Round((sumBlue / matrixWeight) + offset);
                    sumGreen = Math.Round((sumGreen / matrixWeight) + offset);
                    sumRed = Math.Round((sumRed / matrixWeight) + offset);

                    if (sumBlue < 0)
                        sumBlue = 0;
                    if (sumBlue > 255)
                        sumBlue = 255;

                    if (sumGreen < 0)
                        sumGreen = 0;
                    if (sumGreen > 255)
                        sumGreen = 255;

                    if (sumRed < 0)
                        sumRed = 0;
                    if (sumRed > 255)
                        sumRed = 255;

                    pixel = dataPtr;
                    pixel[0] = (byte)sumBlue;
                    pixel[1] = (byte)sumGreen;
                    pixel[2] = (byte)sumRed;



                    // canto (width-1, 0)
                    sumBlue = matrix[0, 0] * (copyPtr + (width - 2) * nChan)[0]
                            + matrix[0, 1] * (copyPtr + (width - 1) * nChan)[0]
                            + matrix[0, 2] * (copyPtr + (width - 1) * nChan)[0]
                            + matrix[1, 0] * (copyPtr + (width - 2) * nChan)[0]
                            + matrix[1, 1] * (copyPtr + (width - 1) * nChan)[0]
                            + matrix[1, 2] * (copyPtr + (width - 1) * nChan)[0]
                            + matrix[2, 0] * (copyPtr + widthstep + (width - 2) * nChan)[0]
                            + matrix[2, 1] * (copyPtr + widthstep + (width - 1) * nChan)[0]
                            + matrix[2, 2] * (copyPtr + widthstep + (width - 1) * nChan)[0];

                    sumGreen = matrix[0, 0] * (copyPtr + (width - 2) * nChan)[1]
                            + matrix[0, 1] * (copyPtr + (width - 1) * nChan)[1]
                            + matrix[0, 2] * (copyPtr + (width - 1) * nChan)[1]
                            + matrix[1, 0] * (copyPtr + (width - 2) * nChan)[1]
                            + matrix[1, 1] * (copyPtr + (width - 1) * nChan)[1]
                            + matrix[1, 2] * (copyPtr + (width - 1) * nChan)[1]
                            + matrix[2, 0] * (copyPtr + widthstep + (width - 2) * nChan)[1]
                            + matrix[2, 1] * (copyPtr + widthstep + (width - 1) * nChan)[1]
                            + matrix[2, 2] * (copyPtr + widthstep + (width - 1) * nChan)[1];

                    sumRed = matrix[0, 0] * (copyPtr + (width - 2) * nChan)[2]
                            + matrix[0, 1] * (copyPtr + (width - 1) * nChan)[2]
                            + matrix[0, 2] * (copyPtr + (width - 1) * nChan)[2]
                            + matrix[1, 0] * (copyPtr + (width - 2) * nChan)[2]
                            + matrix[1, 1] * (copyPtr + (width - 1) * nChan)[2]
                            + matrix[1, 2] * (copyPtr + (width - 1) * nChan)[2]
                            + matrix[2, 0] * (copyPtr + widthstep + (width - 2) * nChan)[2]
                            + matrix[2, 1] * (copyPtr + widthstep + (width - 1) * nChan)[2]
                            + matrix[2, 2] * (copyPtr + widthstep + (width - 1) * nChan)[2];

                    sumBlue = Math.Round((sumBlue / matrixWeight) + offset);
                    sumGreen = Math.Round((sumGreen / matrixWeight) + offset);
                    sumRed = Math.Round((sumRed / matrixWeight) + offset);

                    if (sumBlue < 0)
                        sumBlue = 0;
                    if (sumBlue > 255)
                        sumBlue = 255;

                    if (sumGreen < 0)
                        sumGreen = 0;
                    if (sumGreen > 255)
                        sumGreen = 255;

                    if (sumRed < 0)
                        sumRed = 0;
                    if (sumRed > 255)
                        sumRed = 255;

                    pixel = dataPtr + (width - 1) * nChan;
                    pixel[0] = (byte)sumBlue;
                    pixel[1] = (byte)sumGreen;
                    pixel[2] = (byte)sumRed;



                    // canto (width-1, height-1)
                    sumBlue = matrix[0, 0] * (copyPtr + (height - 2) * widthstep + (width - 2) * nChan)[0]
                            + matrix[0, 1] * (copyPtr + (height - 2) * widthstep + (width - 1) * nChan)[0]
                            + matrix[0, 2] * (copyPtr + (height - 2) * widthstep + (width - 1) * nChan)[0]
                            + matrix[1, 0] * (copyPtr + (height - 1) * widthstep + (width - 2) * nChan)[0]
                            + matrix[1, 1] * (copyPtr + (height - 1) * widthstep + (width - 1) * nChan)[0]
                            + matrix[1, 2] * (copyPtr + (height - 1) * widthstep + (width - 1) * nChan)[0]
                            + matrix[2, 0] * (copyPtr + (height - 1) * widthstep + (width - 2) * nChan)[0]
                            + matrix[2, 1] * (copyPtr + (height - 1) * widthstep + (width - 1) * nChan)[0]
                            + matrix[2, 2] * (copyPtr + (height - 1) * widthstep + (width - 1) * nChan)[0];

                    sumGreen = matrix[0, 0] * (copyPtr + (height - 2) * widthstep + (width - 2) * nChan)[1]
                            + matrix[0, 1] * (copyPtr + (height - 2) * widthstep + (width - 1) * nChan)[1]
                            + matrix[0, 2] * (copyPtr + (height - 2) * widthstep + (width - 1) * nChan)[1]
                            + matrix[1, 0] * (copyPtr + (height - 1) * widthstep + (width - 2) * nChan)[1]
                            + matrix[1, 1] * (copyPtr + (height - 1) * widthstep + (width - 1) * nChan)[1]
                            + matrix[1, 2] * (copyPtr + (height - 1) * widthstep + (width - 1) * nChan)[1]
                            + matrix[2, 0] * (copyPtr + (height - 1) * widthstep + (width - 2) * nChan)[1]
                            + matrix[2, 1] * (copyPtr + (height - 1) * widthstep + (width - 1) * nChan)[1]
                            + matrix[2, 2] * (copyPtr + (height - 1) * widthstep + (width - 1) * nChan)[1];

                    sumRed = matrix[0, 0] * (copyPtr + (height - 2) * widthstep + (width - 2) * nChan)[2]
                            + matrix[0, 1] * (copyPtr + (height - 2) * widthstep + (width - 1) * nChan)[2]
                            + matrix[0, 2] * (copyPtr + (height - 2) * widthstep + (width - 1) * nChan)[2]
                            + matrix[1, 0] * (copyPtr + (height - 1) * widthstep + (width - 2) * nChan)[2]
                            + matrix[1, 1] * (copyPtr + (height - 1) * widthstep + (width - 1) * nChan)[2]
                            + matrix[1, 2] * (copyPtr + (height - 1) * widthstep + (width - 1) * nChan)[2]
                            + matrix[2, 0] * (copyPtr + (height - 1) * widthstep + (width - 2) * nChan)[2]
                            + matrix[2, 1] * (copyPtr + (height - 1) * widthstep + (width - 1) * nChan)[2]
                            + matrix[2, 2] * (copyPtr + (height - 1) * widthstep + (width - 1) * nChan)[2];

                    sumBlue = Math.Round((sumBlue / matrixWeight) + offset);
                    sumGreen = Math.Round((sumGreen / matrixWeight) + offset);
                    sumRed = Math.Round((sumRed / matrixWeight) + offset);

                    if (sumBlue < 0)
                        sumBlue = 0;
                    if (sumBlue > 255)
                        sumBlue = 255;

                    if (sumGreen < 0)
                        sumGreen = 0;
                    if (sumGreen > 255)
                        sumGreen = 255;

                    if (sumRed < 0)
                        sumRed = 0;
                    if (sumRed > 255)
                        sumRed = 255;

                    pixel = dataPtr + (height - 1) * widthstep + (width - 1) * nChan;
                    pixel[0] = (byte)sumBlue;
                    pixel[1] = (byte)sumGreen;
                    pixel[2] = (byte)sumRed;



                    // canto (0, height-1)
                    sumBlue = matrix[0, 0] * (copyPtr + (height - 2) * widthstep)[0]
                            + matrix[0, 1] * (copyPtr + (height - 2) * widthstep)[0]
                            + matrix[0, 2] * (copyPtr + (height - 2) * widthstep + nChan)[0]
                            + matrix[1, 0] * (copyPtr + (height - 1) * widthstep)[0]
                            + matrix[1, 1] * (copyPtr + (height - 1) * widthstep)[0]
                            + matrix[1, 2] * (copyPtr + (height - 1) * widthstep + nChan)[0]
                            + matrix[2, 0] * (copyPtr + (height - 1) * widthstep)[0]
                            + matrix[2, 1] * (copyPtr + (height - 1) * widthstep)[0]
                            + matrix[2, 2] * (copyPtr + (height - 1) * widthstep + nChan)[0];

                    sumGreen = matrix[0, 0] * (copyPtr + (height - 2) * widthstep)[1]
                            + matrix[0, 1] * (copyPtr + (height - 2) * widthstep)[1]
                            + matrix[0, 2] * (copyPtr + (height - 2) * widthstep + nChan)[1]
                            + matrix[1, 0] * (copyPtr + (height - 1) * widthstep)[1]
                            + matrix[1, 1] * (copyPtr + (height - 1) * widthstep)[1]
                            + matrix[1, 2] * (copyPtr + (height - 1) * widthstep + nChan)[1]
                            + matrix[2, 0] * (copyPtr + (height - 1) * widthstep)[1]
                            + matrix[2, 1] * (copyPtr + (height - 1) * widthstep)[1]
                            + matrix[2, 2] * (copyPtr + (height - 1) * widthstep + nChan)[1];

                    sumRed = matrix[0, 0] * (copyPtr + (height - 2) * widthstep)[2]
                            + matrix[0, 1] * (copyPtr + (height - 2) * widthstep)[2]
                            + matrix[0, 2] * (copyPtr + (height - 2) * widthstep + nChan)[2]
                            + matrix[1, 0] * (copyPtr + (height - 1) * widthstep)[2]
                            + matrix[1, 1] * (copyPtr + (height - 1) * widthstep)[2]
                            + matrix[1, 2] * (copyPtr + (height - 1) * widthstep + nChan)[2]
                            + matrix[2, 0] * (copyPtr + (height - 1) * widthstep)[2]
                            + matrix[2, 1] * (copyPtr + (height - 1) * widthstep)[2]
                            + matrix[2, 2] * (copyPtr + (height - 1) * widthstep + nChan)[2];

                    sumBlue = Math.Round((sumBlue / matrixWeight) + offset);
                    sumGreen = Math.Round((sumGreen / matrixWeight) + offset);
                    sumRed = Math.Round((sumRed / matrixWeight) + offset);

                    if (sumBlue < 0)
                        sumBlue = 0;
                    if (sumBlue > 255)
                        sumBlue = 255;

                    if (sumGreen < 0)
                        sumGreen = 0;
                    if (sumGreen > 255)
                        sumGreen = 255;

                    if (sumRed < 0)
                        sumRed = 0;
                    if (sumRed > 255)
                        sumRed = 255;

                    pixel = dataPtr + (height - 1) * widthstep;
                    pixel[0] = (byte)sumBlue;
                    pixel[1] = (byte)sumGreen;
                    pixel[2] = (byte)sumRed;

                }

            }
        }


        /// <summary>
        /// Filtro de sobel
        /// Direct access to memory - faster method
        /// </summary>
        /// <param name="img">Image</param>
        public static void Sobel(Image<Bgr, byte> img, Image<Bgr, byte> imgCopy)
        {
            unsafe
            {
                MIplImage m = img.MIplImage;
                byte* dataPtr = (byte*)m.ImageData.ToPointer();
                MIplImage mCopy = imgCopy.MIplImage;
                byte* copyPtr = (byte*)mCopy.ImageData.ToPointer();

                int x, y;
                int height = img.Height;
                int width = img.Width;
                int nChan = m.NChannels; // number of channels - 3
                int widthstep = m.WidthStep;
                int padding = widthstep - m.Width * m.NChannels;
                byte* pixel;
                double sumBlue, sumGreen, sumRed;
                double sx, sy;

                if (nChan == 3)
                {

                    for (y = 1; y < height - 1; y++)
                    {

                        for (x = 1; x < width - 1; x++)
                        {

                            sx = ((copyPtr + (y - 1) * widthstep + (x - 1) * nChan)[0]
                                + 2 * (copyPtr + y * widthstep + (x - 1) * nChan)[0]
                                + (copyPtr + (y + 1) * widthstep + (x - 1) * nChan)[0])
                                -
                                ((copyPtr + (y - 1) * widthstep + (x + 1) * nChan)[0]
                                + 2 * (copyPtr + y * widthstep + (x + 1) * nChan)[0]
                                + (copyPtr + (y + 1) * widthstep + (x + 1) * nChan)[0]);

                            sy = ((copyPtr + (y + 1) * widthstep + (x - 1) * nChan)[0]
                                + 2 * (copyPtr + (y + 1) * widthstep + x * nChan)[0]
                                + (copyPtr + (y + 1) * widthstep + (x + 1) * nChan)[0])
                                -
                                ((copyPtr + (y - 1) * widthstep + (x - 1) * nChan)[0]
                                + 2 * (copyPtr + (y - 1) * widthstep + x * nChan)[0]
                                + (copyPtr + (y - 1) * widthstep + (x + 1) * nChan)[0]);

                            sumBlue = Math.Round(Math.Abs(sx) + Math.Abs(sy));


                            sx = ((copyPtr + (y - 1) * widthstep + (x - 1) * nChan)[1]
                                + 2 * (copyPtr + y * widthstep + (x - 1) * nChan)[1]
                                + (copyPtr + (y + 1) * widthstep + (x - 1) * nChan)[1])
                                -
                                ((copyPtr + (y - 1) * widthstep + (x + 1) * nChan)[1]
                                + 2 * (copyPtr + y * widthstep + (x + 1) * nChan)[1]
                                + (copyPtr + (y + 1) * widthstep + (x + 1) * nChan)[1]);

                            sy = ((copyPtr + (y + 1) * widthstep + (x - 1) * nChan)[1]
                                + 2 * (copyPtr + (y + 1) * widthstep + x * nChan)[1]
                                + (copyPtr + (y + 1) * widthstep + (x + 1) * nChan)[1])
                                -
                                ((copyPtr + (y - 1) * widthstep + (x - 1) * nChan)[1]
                                + 2 * (copyPtr + (y - 1) * widthstep + x * nChan)[1]
                                + (copyPtr + (y - 1) * widthstep + (x + 1) * nChan)[1]);

                            sumGreen = Math.Round(Math.Abs(sx) + Math.Abs(sy));


                            sx = ((copyPtr + (y - 1) * widthstep + (x - 1) * nChan)[2]
                                + 2 * (copyPtr + y * widthstep + (x - 1) * nChan)[2]
                                + (copyPtr + (y + 1) * widthstep + (x - 1) * nChan)[2])
                                -
                                ((copyPtr + (y - 1) * widthstep + (x + 1) * nChan)[2]
                                + 2 * (copyPtr + y * widthstep + (x + 1) * nChan)[2]
                                + (copyPtr + (y + 1) * widthstep + (x + 1) * nChan)[2]);

                            sy = ((copyPtr + (y + 1) * widthstep + (x - 1) * nChan)[2]
                                + 2 * (copyPtr + (y + 1) * widthstep + x * nChan)[2]
                                + (copyPtr + (y + 1) * widthstep + (x + 1) * nChan)[2])
                                -
                                ((copyPtr + (y - 1) * widthstep + (x - 1) * nChan)[2]
                                + 2 * (copyPtr + (y - 1) * widthstep + x * nChan)[2]
                                + (copyPtr + (y - 1) * widthstep + (x + 1) * nChan)[2]);

                            sumRed = Math.Round(Math.Abs(sx) + Math.Abs(sy));

                            if (sumBlue > 255)
                                sumBlue = 255;

                            if (sumGreen > 255)
                                sumGreen = 255;

                            if (sumRed > 255)
                                sumRed = 255;

                            pixel = dataPtr + y * widthstep + x * nChan;
                            pixel[0] = (byte)sumBlue;
                            pixel[1] = (byte)sumGreen;
                            pixel[2] = (byte)sumRed;
                        }

                    }


                    // primeira linha, sem os cantos
                    for (x = 1; x < width - 1; x++)
                    {
                        sx = 3 * (copyPtr + (x - 1) * nChan)[0]
                            + (copyPtr + widthstep + (x - 1) * nChan)[0]
                            -
                            (3 * (copyPtr + (x + 1) * nChan)[0]
                            + (copyPtr + widthstep + (x + 1) * nChan)[0]);

                        sy = (copyPtr + widthstep + (x - 1) * nChan)[0]
                            + 2 * (copyPtr + widthstep + x * nChan)[0]
                            + (copyPtr + widthstep + (x + 1) * nChan)[0]
                            -
                            ((copyPtr + (x - 1) * nChan)[0]
                            + 2 * (copyPtr + x * nChan)[0]
                            + (copyPtr + (x + 1) * nChan)[0]);

                        sumBlue = Math.Abs(sx) + Math.Abs(sy);


                        sx = 3 * (copyPtr + (x - 1) * nChan)[1]
                            + (copyPtr + widthstep + (x - 1) * nChan)[1]
                            -
                            (3 * (copyPtr + (x + 1) * nChan)[1]
                            + (copyPtr + widthstep + (x + 1) * nChan)[1]);

                        sy = (copyPtr + widthstep + (x - 1) * nChan)[1]
                            + 2 * (copyPtr + widthstep + x * nChan)[1]
                            + (copyPtr + widthstep + (x + 1) * nChan)[1]
                            -
                            ((copyPtr + (x - 1) * nChan)[1]
                            + 2 * (copyPtr + x * nChan)[1]
                            + (copyPtr + (x + 1) * nChan)[1]);

                        sumGreen = Math.Abs(sx) + Math.Abs(sy);


                        sx = 3 * (copyPtr + (x - 1) * nChan)[2]
                            + (copyPtr + widthstep + (x - 1) * nChan)[2]
                            -
                            (3 * (copyPtr + (x + 1) * nChan)[2]
                            + (copyPtr + widthstep + (x + 1) * nChan)[2]);

                        sy = (copyPtr + widthstep + (x - 1) * nChan)[2]
                            + 2 * (copyPtr + widthstep + x * nChan)[2]
                            + (copyPtr + widthstep + (x + 1) * nChan)[2]
                            -
                            ((copyPtr + (x - 1) * nChan)[2]
                            + 2 * (copyPtr + x * nChan)[2]
                            + (copyPtr + (x + 1) * nChan)[2]);

                        sumRed = Math.Abs(sx) + Math.Abs(sy);

                        if (sumBlue > 255)
                            sumBlue = 255;

                        if (sumGreen > 255)
                            sumGreen = 255;

                        if (sumRed > 255)
                            sumRed = 255;

                        pixel = dataPtr + x * nChan;
                        pixel[0] = (byte)sumBlue;
                        pixel[1] = (byte)sumGreen;
                        pixel[2] = (byte)sumRed;

                    }



                    // ultima linha, sem os cantos
                    for (x = 1; x < width - 1; x++)
                    {
                        sx = (copyPtr + (height - 2) * widthstep + (x - 1) * nChan)[0]
                            + 3 * (copyPtr + (height - 1) * widthstep + (x - 1) * nChan)[0]
                            -
                            ((copyPtr + (height - 2) * widthstep + (x + 1) * nChan)[0]
                            + 3 * (copyPtr + (height - 1) * widthstep + (x + 1) * nChan)[0]);

                        sy = (copyPtr + (height - 1) * widthstep + (x - 1) * nChan)[0]
                            + 2 * (copyPtr + (height - 1) * widthstep + x * nChan)[0]
                            + (copyPtr + (height - 1) * widthstep + (x + 1) * nChan)[0]
                            -
                            ((copyPtr + (height - 2) * widthstep + (x - 1) * nChan)[0]
                            + 2 * (copyPtr + (height - 2) * widthstep + x * nChan)[0]
                            + (copyPtr + (height - 2) * widthstep + (x + 1) * nChan)[0]);

                        sumBlue = Math.Abs(sx) + Math.Abs(sy);


                        sx = (copyPtr + (height - 2) * widthstep + (x - 1) * nChan)[1]
                            + 3 * (copyPtr + (height - 1) * widthstep + (x - 1) * nChan)[1]
                            -
                            ((copyPtr + (height - 2) * widthstep + (x + 1) * nChan)[1]
                            + 3 * (copyPtr + (height - 1) * widthstep + (x + 1) * nChan)[1]);

                        sy = (copyPtr + (height - 1) * widthstep + (x - 1) * nChan)[1]
                            + 2 * (copyPtr + (height - 1) * widthstep + x * nChan)[1]
                            + (copyPtr + (height - 1) * widthstep + (x + 1) * nChan)[1]
                            -
                            ((copyPtr + (height - 2) * widthstep + (x - 1) * nChan)[1]
                            + 2 * (copyPtr + (height - 2) * widthstep + x * nChan)[1]
                            + (copyPtr + (height - 2) * widthstep + (x + 1) * nChan)[1]);

                        sumGreen = Math.Abs(sx) + Math.Abs(sy);


                        sx = (copyPtr + (height - 2) * widthstep + (x - 1) * nChan)[2]
                            + 3 * (copyPtr + (height - 1) * widthstep + (x - 1) * nChan)[2]
                            -
                            ((copyPtr + (height - 2) * widthstep + (x + 1) * nChan)[2]
                            + 3 * (copyPtr + (height - 1) * widthstep + (x + 1) * nChan)[2]);

                        sy = (copyPtr + (height - 1) * widthstep + (x - 1) * nChan)[2]
                            + 2 * (copyPtr + (height - 1) * widthstep + x * nChan)[2]
                            + (copyPtr + (height - 1) * widthstep + (x + 1) * nChan)[2]
                            -
                            ((copyPtr + (height - 2) * widthstep + (x - 1) * nChan)[2]
                            + 2 * (copyPtr + (height - 2) * widthstep + x * nChan)[2]
                            + (copyPtr + (height - 2) * widthstep + (x + 1) * nChan)[2]);

                        sumRed = Math.Abs(sx) + Math.Abs(sy);

                        if (sumBlue > 255)
                            sumBlue = 255;

                        if (sumGreen > 255)
                            sumGreen = 255;

                        if (sumRed > 255)
                            sumRed = 255;

                        pixel = dataPtr + (height - 1) * widthstep + x * nChan;
                        pixel[0] = (byte)sumBlue;
                        pixel[1] = (byte)sumGreen;
                        pixel[2] = (byte)sumRed;

                    }



                    // primeira coluna, sem os cantos
                    for (y = 1; y < height - 1; y++)
                    {
                        sx = (copyPtr + (y - 1) * widthstep)[0]
                            + 2 * (copyPtr + y * widthstep)[0]
                            + (copyPtr + (y + 1) * widthstep)[0]
                            -
                            ((copyPtr + (y - 1) * widthstep + nChan)[0]
                            + 2 * (copyPtr + y * widthstep + nChan)[0]
                            + (copyPtr + (y + 1) * widthstep + nChan)[0]);

                        sy = 3 * (copyPtr + (y + 1) * widthstep)[0]
                            + (copyPtr + (y + 1) * widthstep + nChan)[0]
                            -
                            (3 * (copyPtr + (y - 1) * widthstep)[0]
                            + (copyPtr + (y - 1) * widthstep + nChan)[0]);

                        sumBlue = Math.Abs(sx) + Math.Abs(sy);


                        sx = (copyPtr + (y - 1) * widthstep)[1]
                            + 2 * (copyPtr + y * widthstep)[1]
                            + (copyPtr + (y + 1) * widthstep)[1]
                            -
                            ((copyPtr + (y - 1) * widthstep + nChan)[1]
                            + 2 * (copyPtr + y * widthstep + nChan)[1]
                            + (copyPtr + (y + 1) * widthstep + nChan)[1]);

                        sy = 3 * (copyPtr + (y + 1) * widthstep)[1]
                            + (copyPtr + (y + 1) * widthstep + nChan)[1]
                            -
                            (3 * (copyPtr + (y - 1) * widthstep)[1]
                            + (copyPtr + (y - 1) * widthstep + nChan)[1]);

                        sumGreen = Math.Abs(sx) + Math.Abs(sy);


                        sx = (copyPtr + (y - 1) * widthstep)[2]
                            + 2 * (copyPtr + y * widthstep)[2]
                            + (copyPtr + (y + 1) * widthstep)[2]
                            -
                            ((copyPtr + (y - 1) * widthstep + nChan)[2]
                            + 2 * (copyPtr + y * widthstep + nChan)[2]
                            + (copyPtr + (y + 1) * widthstep + nChan)[2]);

                        sy = 3 * (copyPtr + (y + 1) * widthstep)[2]
                            + (copyPtr + (y + 1) * widthstep + nChan)[2]
                            -
                            (3 * (copyPtr + (y - 1) * widthstep)[2]
                            + (copyPtr + (y - 1) * widthstep + nChan)[2]);

                        sumRed = Math.Abs(sx) + Math.Abs(sy);


                        if (sumBlue > 255)
                            sumBlue = 255;

                        if (sumGreen > 255)
                            sumGreen = 255;

                        if (sumRed > 255)
                            sumRed = 255;

                        pixel = dataPtr + y * widthstep;
                        pixel[0] = (byte)sumBlue;
                        pixel[1] = (byte)sumGreen;
                        pixel[2] = (byte)sumRed;

                    }



                    // ultima coluna, sem os cantos
                    for (y = 1; y < height - 1; y++)
                    {
                        sx = (copyPtr + (y - 1) * widthstep + (width - 2) * nChan)[0]
                            + 2 * (copyPtr + y * widthstep + (width - 2) * nChan)[0]
                            + (copyPtr + (y + 1) * widthstep + (width - 2) * nChan)[0]
                            -
                            ((copyPtr + (y - 1) * widthstep + (width - 1) * nChan)[0]
                            + 2 * (copyPtr + y * widthstep + (width - 1) * nChan)[0]
                            + (copyPtr + (y + 1) * widthstep + (width - 1) * nChan)[0]);

                        sy = 3 * (copyPtr + (y + 1) * widthstep + (width - 1) * nChan)[0]
                            + (copyPtr + (y + 1) * widthstep + (width - 2) * nChan)[0]
                            -
                            (3 * (copyPtr + (y - 1) * widthstep + (width - 1) * nChan)[0]
                            + (copyPtr + (y - 1) * widthstep + (width - 2) * nChan)[0]);

                        sumBlue = Math.Abs(sx) + Math.Abs(sy);


                        sx = (copyPtr + (y - 1) * widthstep + (width - 2) * nChan)[1]
                            + 2 * (copyPtr + y * widthstep + (width - 2) * nChan)[1]
                            + (copyPtr + (y + 1) * widthstep + (width - 2) * nChan)[1]
                            -
                            ((copyPtr + (y - 1) * widthstep + (width - 1) * nChan)[1]
                            + 2 * (copyPtr + y * widthstep + (width - 1) * nChan)[1]
                            + (copyPtr + (y + 1) * widthstep + (width - 1) * nChan)[1]);

                        sy = 3 * (copyPtr + (y + 1) * widthstep + (width - 1) * nChan)[1]
                            + (copyPtr + (y + 1) * widthstep + (width - 2) * nChan)[1]
                            -
                            (3 * (copyPtr + (y - 1) * widthstep + (width - 1) * nChan)[1]
                            + (copyPtr + (y - 1) * widthstep + (width - 2) * nChan)[1]);

                        sumGreen = Math.Abs(sx) + Math.Abs(sy);


                        sx = (copyPtr + (y - 1) * widthstep + (width - 2) * nChan)[2]
                            + 2 * (copyPtr + y * widthstep + (width - 2) * nChan)[2]
                            + (copyPtr + (y + 1) * widthstep + (width - 2) * nChan)[2]
                            -
                            ((copyPtr + (y - 1) * widthstep + (width - 1) * nChan)[2]
                            + 2 * (copyPtr + y * widthstep + (width - 1) * nChan)[2]
                            + (copyPtr + (y + 1) * widthstep + (width - 1) * nChan)[2]);

                        sy = 3 * (copyPtr + (y + 1) * widthstep + (width - 1) * nChan)[2]
                            + (copyPtr + (y + 1) * widthstep + (width - 2) * nChan)[2]
                            -
                            (3 * (copyPtr + (y - 1) * widthstep + (width - 1) * nChan)[2]
                            + (copyPtr + (y - 1) * widthstep + (width - 2) * nChan)[2]);

                        sumRed = Math.Abs(sx) + Math.Abs(sy);


                        if (sumBlue > 255)
                            sumBlue = 255;

                        if (sumGreen > 255)
                            sumGreen = 255;

                        if (sumRed > 255)
                            sumRed = 255;

                        pixel = dataPtr + y * widthstep + (width - 1) * nChan;
                        pixel[0] = (byte)sumBlue;
                        pixel[1] = (byte)sumGreen;
                        pixel[2] = (byte)sumRed;

                    }


                    // canto (0,0)
                    sx = 3 * (copyPtr)[0]
                        + (copyPtr + widthstep)[0]
                        -
                        (3 * (copyPtr + nChan)[0]
                        + (copyPtr + widthstep + nChan)[0]);

                    sy = 3 * (copyPtr + widthstep)[0]
                        + (copyPtr + widthstep + nChan)[0]
                        -
                        (3 * (copyPtr)[0]
                        + (copyPtr + nChan)[0]);

                    sumBlue = Math.Abs(sx) + Math.Abs(sy);


                    sx = 3 * (copyPtr)[1]
                        + (copyPtr + widthstep)[1]
                        -
                        (3 * (copyPtr + nChan)[1]
                        + (copyPtr + widthstep + nChan)[1]);

                    sy = 3 * (copyPtr + widthstep)[1]
                        + (copyPtr + widthstep + nChan)[1]
                        -
                        (3 * (copyPtr)[1]
                        + (copyPtr + nChan)[1]);

                    sumGreen = Math.Abs(sx) + Math.Abs(sy);


                    sx = 3 * (copyPtr)[2]
                        + (copyPtr + widthstep)[2]
                        -
                        (3 * (copyPtr + nChan)[2]
                        + (copyPtr + widthstep + nChan)[2]);

                    sy = 3 * (copyPtr + widthstep)[2]
                        + (copyPtr + widthstep + nChan)[2]
                        -
                        (3 * (copyPtr)[2]
                        + (copyPtr + nChan)[2]);

                    sumRed = Math.Abs(sx) + Math.Abs(sy);


                    if (sumBlue > 255)
                        sumBlue = 255;

                    if (sumGreen > 255)
                        sumGreen = 255;

                    if (sumRed > 255)
                        sumRed = 255;

                    pixel = dataPtr;
                    pixel[0] = (byte)sumBlue;
                    pixel[1] = (byte)sumGreen;
                    pixel[2] = (byte)sumRed;


                    // canto (width - 1, 0)
                    sx = 3 * (copyPtr + (width - 2) * nChan)[0]
                        + (copyPtr + widthstep + (width - 2) * nChan)[0]
                        -
                        (3 * (copyPtr + (width - 1) * nChan)[0]
                        + (copyPtr + widthstep + (width - 1) * nChan)[0]);

                    sy = 3 * (copyPtr + widthstep + (width - 1) * nChan)[0]
                        + (copyPtr + widthstep + (width - 2) * nChan)[0]
                        -
                        (3 * (copyPtr + (width - 1) * nChan)[0]
                        + (copyPtr + (width - 2) * nChan)[0]);

                    sumBlue = Math.Abs(sx) + Math.Abs(sy);


                    sx = 3 * (copyPtr + (width - 2) * nChan)[1]
                        + (copyPtr + widthstep + (width - 2) * nChan)[1]
                        -
                        (3 * (copyPtr + (width - 1) * nChan)[1]
                        + (copyPtr + widthstep + (width - 1) * nChan)[1]);

                    sy = 3 * (copyPtr + widthstep + (width - 1) * nChan)[1]
                        + (copyPtr + widthstep + (width - 2) * nChan)[1]
                        -
                        (3 * (copyPtr + (width - 1) * nChan)[1]
                        + (copyPtr + (width - 2) * nChan)[1]);

                    sumGreen = Math.Abs(sx) + Math.Abs(sy);


                    sx = 3 * (copyPtr + (width - 2) * nChan)[2]
                        + (copyPtr + widthstep + (width - 2) * nChan)[2]
                        -
                        (3 * (copyPtr + (width - 1) * nChan)[2]
                        + (copyPtr + widthstep + (width - 1) * nChan)[2]);

                    sy = 3 * (copyPtr + widthstep + (width - 1) * nChan)[2]
                        + (copyPtr + widthstep + (width - 2) * nChan)[2]
                        -
                        (3 * (copyPtr + (width - 1) * nChan)[2]
                        + (copyPtr + (width - 2) * nChan)[2]);

                    sumRed = Math.Abs(sx) + Math.Abs(sy);


                    if (sumBlue > 255)
                        sumBlue = 255;

                    if (sumGreen > 255)
                        sumGreen = 255;

                    if (sumRed > 255)
                        sumRed = 255;

                    pixel = dataPtr + (width - 1) * nChan;
                    pixel[0] = (byte)sumBlue;
                    pixel[1] = (byte)sumGreen;
                    pixel[2] = (byte)sumRed;



                    // canto (0, height - 1)
                    sx = 3 * (copyPtr + (height - 1) * widthstep)[0]
                        + (copyPtr + (height - 2) * widthstep)[0]
                        -
                        (3 * (copyPtr + (height - 1) * widthstep + nChan)[0]
                        + (copyPtr + (height - 2) * widthstep + nChan)[0]);

                    sy = 3 * (copyPtr + (height - 1) * widthstep)[0]
                        + (copyPtr + (height - 1) * widthstep + nChan)[0]
                        -
                        (3 * (copyPtr + (height - 2) * widthstep)[0]
                        + (copyPtr + (height - 2) * widthstep + nChan)[0]);

                    sumBlue = Math.Abs(sx) + Math.Abs(sy);


                    sx = 3 * (copyPtr + (height - 1) * widthstep)[1]
                        + (copyPtr + (height - 2) * widthstep)[1]
                        -
                        (3 * (copyPtr + (height - 1) * widthstep + nChan)[1]
                        + (copyPtr + (height - 2) * widthstep + nChan)[1]);

                    sy = 3 * (copyPtr + (height - 1) * widthstep)[1]
                        + (copyPtr + (height - 1) * widthstep + nChan)[1]
                        -
                        (3 * (copyPtr + (height - 2) * widthstep)[1]
                        + (copyPtr + (height - 2) * widthstep + nChan)[1]);

                    sumGreen = Math.Abs(sx) + Math.Abs(sy);


                    sx = 3 * (copyPtr + (height - 1) * widthstep)[2]
                        + (copyPtr + (height - 2) * widthstep)[2]
                        -
                        (3 * (copyPtr + (height - 1) * widthstep + nChan)[2]
                        + (copyPtr + (height - 2) * widthstep + nChan)[2]);

                    sy = 3 * (copyPtr + (height - 1) * widthstep)[2]
                        + (copyPtr + (height - 1) * widthstep + nChan)[2]
                        -
                        (3 * (copyPtr + (height - 2) * widthstep)[2]
                        + (copyPtr + (height - 2) * widthstep + nChan)[2]);

                    sumRed = Math.Abs(sx) + Math.Abs(sy);


                    if (sumBlue > 255)
                        sumBlue = 255;

                    if (sumGreen > 255)
                        sumGreen = 255;

                    if (sumRed > 255)
                        sumRed = 255;

                    pixel = dataPtr + (height - 1) * widthstep;
                    pixel[0] = (byte)sumBlue;
                    pixel[1] = (byte)sumGreen;
                    pixel[2] = (byte)sumRed;



                    // canto (width - 1, height - 1)
                    sx = 3 * (copyPtr + (height - 1) * widthstep + (width - 2) * nChan)[0]
                        + (copyPtr + (height - 2) * widthstep + (width - 2) * nChan)[0]
                        -
                        (3 * (copyPtr + (height - 1) * widthstep + (width - 1) * nChan)[0]
                        + (copyPtr + (height - 2) * widthstep + (width - 1) * nChan)[0]);

                    sy = 3 * (copyPtr + (height - 1) * widthstep + (width - 1) * nChan)[0]
                        + (copyPtr + (height - 1) * widthstep + (width - 2) * nChan)[0]
                        -
                        (3 * (copyPtr + (height - 2) * widthstep + (width - 1) * nChan)[0]
                        + (copyPtr + (height - 2) * widthstep + (width - 2) * nChan)[0]);

                    sumBlue = Math.Abs(sx) + Math.Abs(sy);


                    sx = 3 * (copyPtr + (height - 1) * widthstep + (width - 2) * nChan)[1]
                        + (copyPtr + (height - 2) * widthstep + (width - 2) * nChan)[1]
                        -
                        (3 * (copyPtr + (height - 1) * widthstep + (width - 1) * nChan)[1]
                        + (copyPtr + (height - 2) * widthstep + (width - 1) * nChan)[1]);

                    sy = 3 * (copyPtr + (height - 1) * widthstep + (width - 1) * nChan)[1]
                        + (copyPtr + (height - 1) * widthstep + (width - 2) * nChan)[1]
                        -
                        (3 * (copyPtr + (height - 2) * widthstep + (width - 1) * nChan)[1]
                        + (copyPtr + (height - 2) * widthstep + (width - 2) * nChan)[1]);

                    sumGreen = Math.Abs(sx) + Math.Abs(sy);


                    sx = 3 * (copyPtr + (height - 1) * widthstep + (width - 2) * nChan)[2]
                        + (copyPtr + (height - 2) * widthstep + (width - 2) * nChan)[2]
                        -
                        (3 * (copyPtr + (height - 1) * widthstep + (width - 1) * nChan)[2]
                        + (copyPtr + (height - 2) * widthstep + (width - 1) * nChan)[2]);

                    sy = 3 * (copyPtr + (height - 1) * widthstep + (width - 1) * nChan)[2]
                        + (copyPtr + (height - 1) * widthstep + (width - 2) * nChan)[2]
                        -
                        (3 * (copyPtr + (height - 2) * widthstep + (width - 1) * nChan)[2]
                        + (copyPtr + (height - 2) * widthstep + (width - 2) * nChan)[2]);

                    sumRed = Math.Abs(sx) + Math.Abs(sy);


                    if (sumBlue > 255)
                        sumBlue = 255;

                    if (sumGreen > 255)
                        sumGreen = 255;

                    if (sumRed > 255)
                        sumRed = 255;

                    pixel = dataPtr + (height - 1) * widthstep + (width - 1) * nChan;
                    pixel[0] = (byte)sumBlue;
                    pixel[1] = (byte)sumGreen;
                    pixel[2] = (byte)sumRed;

                }
            }
        }


        /// <summary>
        /// Filtro de diferenciação
        /// Direct access to memory - faster method
        /// </summary>
        /// <param name="img">Image</param>
        public static void Diferentiation(Image<Bgr, byte> img, Image<Bgr, byte> imgCopy)
        {
            unsafe
            {
                MIplImage m = img.MIplImage;
                byte* dataPtr = (byte*)m.ImageData.ToPointer();
                MIplImage mCopy = imgCopy.MIplImage;
                byte* copyPtr = (byte*)mCopy.ImageData.ToPointer();

                int x, y;
                int height = img.Height;
                int width = img.Width;
                int nChan = m.NChannels; // number of channels - 3
                int widthstep = m.WidthStep;
                int padding = widthstep - m.Width * m.NChannels;
                byte* pixel;
                double sumBlue, sumGreen, sumRed;
                double sx, sy;

                if (nChan == 3)
                {

                    for (y = 0; y < height - 1; y++)
                    {

                        for (x = 0; x < width - 1; x++)
                        {

                            sx = (copyPtr + y * widthstep + x * nChan)[0] - (copyPtr + y * widthstep + (x + 1) * nChan)[0];
                            sy = (copyPtr + y * widthstep + x * nChan)[0] - (copyPtr + (y + 1) * widthstep + x * nChan)[0];
                            sumBlue = Math.Abs(sx) + Math.Abs(sy);

                            sx = (copyPtr + y * widthstep + x * nChan)[1] - (copyPtr + y * widthstep + (x + 1) * nChan)[1];
                            sy = (copyPtr + y * widthstep + x * nChan)[1] - (copyPtr + (y + 1) * widthstep + x * nChan)[1];
                            sumGreen = Math.Abs(sx) + Math.Abs(sy);

                            sx = (copyPtr + y * widthstep + x * nChan)[2] - (copyPtr + y * widthstep + (x + 1) * nChan)[2];
                            sy = (copyPtr + y * widthstep + x * nChan)[2] - (copyPtr + (y + 1) * widthstep + x * nChan)[2];
                            sumRed = Math.Abs(sx) + Math.Abs(sy);


                            if (sumBlue > 255)
                                sumBlue = 255;

                            if (sumGreen > 255)
                                sumGreen = 255;

                            if (sumRed > 255)
                                sumRed = 255;

                            pixel = dataPtr + y * widthstep + x * nChan;
                            pixel[0] = (byte)sumBlue;
                            pixel[1] = (byte)sumGreen;
                            pixel[2] = (byte)sumRed;
                        }

                    }

                    // ultima linha, sem o canto direito
                    for (x = 0; x < width - 1; x++)
                    {

                        sx = (copyPtr + y * widthstep + x * nChan)[0] - (copyPtr + y * widthstep + (x + 1) * nChan)[0];
                        // sy seria o pixel subtraido dele proprio, logo é 0
                        sumBlue = Math.Abs(sx);

                        sx = (copyPtr + y * widthstep + x * nChan)[1] - (copyPtr + y * widthstep + (x + 1) * nChan)[1];
                        sumGreen = Math.Abs(sx);

                        sx = (copyPtr + y * widthstep + x * nChan)[2] - (copyPtr + y * widthstep + (x + 1) * nChan)[2];
                        sumRed = Math.Abs(sx);

                        if (sumBlue > 255)
                            sumBlue = 255;

                        if (sumGreen > 255)
                            sumGreen = 255;

                        if (sumRed > 255)
                            sumRed = 255;

                        pixel = dataPtr + (height - 1) * widthstep + x * nChan;
                        pixel[0] = (byte)sumBlue;
                        pixel[1] = (byte)sumGreen;
                        pixel[2] = (byte)sumRed;

                    }


                    // ultima coluna, sem o canto inferior
                    for (y = 0; y < height - 1; y++)
                    {

                        sy = (copyPtr + y * widthstep + x * nChan)[0] - (copyPtr + (y + 1) * widthstep + x * nChan)[0];
                        // sx seria o pixel subtraido dele proprio, logo é 0
                        sumBlue = Math.Abs(sy);

                        sy = (copyPtr + y * widthstep + x * nChan)[1] - (copyPtr + (y + 1) * widthstep + x * nChan)[1];
                        sumGreen = Math.Abs(sy);

                        sy = (copyPtr + y * widthstep + x * nChan)[2] - (copyPtr + (y + 1) * widthstep + x * nChan)[2];
                        sumRed = Math.Abs(sy);

                        if (sumBlue > 255)
                            sumBlue = 255;

                        if (sumGreen > 255)
                            sumGreen = 255;

                        if (sumRed > 255)
                            sumRed = 255;

                        pixel = dataPtr + y * widthstep + (width - 1) * nChan;
                        pixel[0] = (byte)sumBlue;
                        pixel[1] = (byte)sumGreen;
                        pixel[2] = (byte)sumRed;

                    }


                    // canto (width - 1, height - 1)
                    // sx e sy sao 0, porque é o pixel subtraido dele proprio
                    pixel = dataPtr + (height - 1) * widthstep + (width - 1) * nChan;
                    pixel[0] = (byte)0;
                    pixel[1] = (byte)0;
                    pixel[2] = (byte)0;

                }
            }
        }

        /// <summary>
        /// Filtro de mediana
        /// Direct access to memory - faster method
        /// </summary>
        /// <param name="img">Image</param>
        public static void Median(Image<Bgr, byte> img, Image<Bgr, byte> imgCopy)
        {
            unsafe
            {
                CvInvoke.MedianBlur(imgCopy, img, 3);
            }
        }


        /// <summary>
        /// Histograma da escala de cinzentos
        /// Direct access to memory - faster method
        /// </summary>
        /// <param name="img">Image</param>
        public static int[] Histogram_Gray(Image<Bgr, byte> img)
        {
            unsafe
            {
                MIplImage origem = img.MIplImage;
                byte* dataPtr_o = (byte*)origem.ImageData.ToPointer(); // Pointer to the image

                int[] lista = new int[256];
                int width = origem.Width;
                int height = origem.Height;
                int widthStep = origem.WidthStep;
                int nChan = origem.NChannels; // number of channels = 3
                int blue, green, red, grey;

                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        blue = (dataPtr_o + y * widthStep + x * nChan)[0];
                        green = (dataPtr_o + y * widthStep + x * nChan)[1];
                        red = (dataPtr_o + y * widthStep + x * nChan)[2];

                        grey = (int)Math.Round(((int)blue + green + red) / 3.0);
                        lista[grey]++;

                    }
                }
                return lista;
            }
        }

        /// <summary>
        /// Histograma das cores
        /// Direct access to memory - faster method
        /// </summary>
        /// <param name="img">Image</param>
        public static int[,] Histogram_RGB(Image<Bgr, byte> img)
        {
            unsafe
            {
                MIplImage m = img.MIplImage;
                byte* dataPtr = (byte*)m.ImageData.ToPointer();

                int[,] hist_bgr = new int[3, 256];
                int width = img.Width;
                int height = img.Height;
                int nChan = m.NChannels; // number of channels - 3
                int padding = m.WidthStep - m.NChannels * m.Width; // alinhament bytes (padding)
                int x, y;
                byte blue, green, red;


                if (nChan == 3) // image in RGB
                {
                    for (y = 0; y < height; y++)
                    {
                        for (x = 0; x < width; x++)
                        {
                            //retrive 3 colour components
                            blue = dataPtr[0];
                            green = dataPtr[1];
                            red = dataPtr[2];

                            // store in the histogram array
                            hist_bgr[0, blue]++;
                            hist_bgr[1, green]++;
                            hist_bgr[2, red]++;


                            // advance the pointer to the next pixel
                            dataPtr += nChan;
                        }

                        //at the end of the line advance the pointer by the aligment bytes (padding)
                        dataPtr += padding;
                    }
                }

                return hist_bgr;
            }
        }

        /// <summary>
        /// Histograma da escala de todos
        /// Direct access to memory - faster method
        /// </summary>
        /// <param name="img">Image</param>
        public static int[,] Histogram_All(Image<Bgr, byte> img)
        {
            unsafe
            {
                MIplImage m = img.MIplImage;
                byte* dataPtr = (byte*)m.ImageData.ToPointer();

                int[,] hist_all = new int[4, 256];
                int width = img.Width;
                int height = img.Height;
                int nChan = m.NChannels; // number of channels - 3
                int padding = m.WidthStep - m.NChannels * m.Width; // alinhament bytes (padding)
                int x, y;
                byte blue, green, red, gray;


                if (nChan == 3) // image in RGB
                {
                    for (y = 0; y < height; y++)
                    {
                        for (x = 0; x < width; x++)
                        {
                            //retrive 3 colour components
                            blue = dataPtr[0];
                            green = dataPtr[1];
                            red = dataPtr[2];

                            gray = (byte)Math.Round(((int)blue + green + red) / 3.0);

                            // store in the histogram array
                            hist_all[0, gray]++;
                            hist_all[1, blue]++;
                            hist_all[2, green]++;
                            hist_all[3, red]++;


                            // advance the pointer to the next pixel
                            dataPtr += nChan;
                        }

                        //at the end of the line advance the pointer by the aligment bytes (padding)
                        dataPtr += padding;
                    }
                }

                return hist_all;
            }
        }

        /// <summary>
        /// Binarização
        /// </summary>
        /// <param name="img">Image</param>
        public static void ConvertToBW(Image<Bgr, byte> img, int threshold)
        {
            unsafe
            {
                MIplImage origem = img.MIplImage;
                byte* dataPtr_o = (byte*)origem.ImageData.ToPointer(); // Pointer to the image

                int width = origem.Width;
                int height = origem.Height;
                int widthStep = origem.WidthStep;
                int nChan = origem.NChannels; // number of channels = 3
                int blue, green, red, grey;

                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        blue = (dataPtr_o + y * widthStep + x * nChan)[0];
                        green = (dataPtr_o + y * widthStep + x * nChan)[1];
                        red = (dataPtr_o + y * widthStep + x * nChan)[2];

                        grey = (int)Math.Round(((int)blue + green + red) / 3.0);

                        if (grey <= threshold)
                        {
                            grey = 0;
                            (dataPtr_o + y * widthStep + x * nChan)[0] = (byte)grey;
                            (dataPtr_o + y * widthStep + x * nChan)[1] = (byte)grey;
                            (dataPtr_o + y * widthStep + x * nChan)[2] = (byte)grey;
                        }
                        else
                        {
                            grey = 255;
                            (dataPtr_o + y * widthStep + x * nChan)[0] = (byte)grey;
                            (dataPtr_o + y * widthStep + x * nChan)[1] = (byte)grey;
                            (dataPtr_o + y * widthStep + x * nChan)[2] = (byte)grey;
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Binarização OTSU
        /// Direct access to memory - faster method
        /// </summary>
        /// <param name="img">Image</param>
        public static void ConvertToBW_Otsu(Image<Bgr, byte> img)
        {
            unsafe
            {
                MIplImage origem = img.MIplImage;
                byte* dataPtr_o = (byte*)origem.ImageData.ToPointer(); // Pointer to the image

                int width = origem.Width;
                int height = origem.Height;
                int widthStep = origem.WidthStep;
                int nChan = origem.NChannels; // number of channels = 3
                int[] hist = Histogram_Gray(img); //lista com quantidade de tons de cinzento

                int t, i;
                int threshold = 157;
                double max = 0;
                double pixels_totais = width * height;
                double pi, q1, q2, u1, u2, s1, s2, sigma;

                for (t = 0; t < 256; t++)
                {
                    s1 = 0;
                    q1 = 0;
                    s2 = 0;
                    q2 = 0;

                    for (i = 0; i <= t; i++)
                    {
                        pi = hist[i] / pixels_totais; //nº de pixeis com intensidade i/nº total de pixeis
                        q1 += pi; //probabilidade acomulada
                        s1 += i * pi; //intensidade * probabilidade
                    }

                    for (i = t + 1; i < 256; i++)
                    {
                        pi = hist[i] / pixels_totais;
                        q2 += pi;
                        s2 += i * pi;
                    }

                    u1 = s1 / q1; //média ponderada do t - esquerda
                    u2 = s2 / q2; //média ponderada do t - direita

                    sigma = q1 * q2 * (Math.Pow((u1 - u2), 2)); //(u1 - u2)^2

                    if (sigma > max)
                    {
                        max = sigma;
                        threshold = t; //o t que tiver maior variancia é aquele que o método vai retornar
                    }
                }
                ConvertToBW(img, threshold);
            }
        }

        /// <summary>
        /// Roberts
        /// Direct access to memory - faster method
        /// </summary>
        /// <param name="img">Image</param>
        public static void Roberts(Image<Bgr, byte> img, Image<Bgr, byte> imgCopy)
        {
            unsafe
            {
                // direct access to the image memory(sequencial)
                // direcion top left -> bottom right

                MIplImage m = img.MIplImage;
                byte* dataPtr1 = (byte*)m.ImageData.ToPointer(); // Pointer to the image

                MIplImage mC = imgCopy.MIplImage;
                byte* dataPtr2 = (byte*)mC.ImageData.ToPointer(); // Pointer to the image

                int width = img.Width;
                int height = img.Height;
                int nChan = m.NChannels; // number of channels - 3
                int widthStep = m.WidthStep;
                int y = 0;
                int x = 0;

                int padding = m.WidthStep - m.NChannels * m.Width; // alinhament bytes (padding)

                int blue = 0;
                int green = 0;
                int red = 0;

                // pixel --- right
                // below --- below_right

                for (y = 0; y < (height - 1); y++)
                {
                    //core
                    for (x = 0; x < (width - 1); x++)
                    {

                        blue = (int)(Math.Abs(dataPtr2[0] - (dataPtr2 + widthStep + nChan)[0]) + Math.Abs((dataPtr2 + nChan)[0] - (dataPtr2 + widthStep)[0]));

                        green = (int)(Math.Abs(dataPtr2[1] - (dataPtr2 + widthStep + nChan)[1]) + Math.Abs((dataPtr2 + nChan)[1] - (dataPtr2 + widthStep)[1]));

                        red = (int)(Math.Abs(dataPtr2[2] - (dataPtr2 + widthStep + nChan)[2]) + Math.Abs((dataPtr2 + nChan)[2] - (dataPtr2 + widthStep)[2]));

                        if (blue > 255)
                            blue = 255;

                        if (green > 255)
                            green = 255;

                        if (red > 255)
                            red = 255;

                        dataPtr1[0] = (byte)blue;
                        dataPtr1[1] = (byte)green;
                        dataPtr1[2] = (byte)red;

                        dataPtr1 += nChan;
                        dataPtr2 += nChan;
                    }

                    //coluna da direita

                    blue = (int)(2 * Math.Abs(dataPtr2[0] - (dataPtr2 + widthStep)[0]));

                    green = (int)(2 * Math.Abs(dataPtr2[1] - (dataPtr2 + widthStep)[1]));

                    red = (int)(2 * Math.Abs(dataPtr2[2] - (dataPtr2 + widthStep)[2]));

                    if (blue > 255)
                        blue = 255;

                    if (green > 255)
                        green = 255;

                    if (red > 255)
                        red = 255;

                    dataPtr1[0] = (byte)blue;
                    dataPtr1[1] = (byte)green;
                    dataPtr1[2] = (byte)red;


                    dataPtr1 += (padding + nChan);
                    dataPtr2 += (padding + nChan);
                }

                //ultima linha
                for (x = 0; x < (width - 1); x++)
                {

                    blue = (int)(2 * Math.Abs(dataPtr2[0] - (dataPtr2 + nChan)[0]));

                    green = (int)(2 * Math.Abs(dataPtr2[1] - (dataPtr2 + nChan)[1]));

                    red = (int)(2 * Math.Abs(dataPtr2[2] - (dataPtr2 + nChan)[2]));

                    if (blue > 255)
                        blue = 255;

                    if (green > 255)
                        green = 255;

                    if (red > 255)
                        red = 255;

                    dataPtr1[0] = (byte)blue;
                    dataPtr1[1] = (byte)green;
                    dataPtr1[2] = (byte)red;

                    dataPtr1 += nChan;
                    dataPtr2 += nChan;
                }

                //canto inferior direito

                dataPtr1[0] = (byte)(0);
                dataPtr1[1] = (byte)(0);
                dataPtr1[2] = (byte)(0);


            }
        }
        /*************************************TRABALHO FINAL*************************************/

        public static void Dilatation(Image<Bgr, byte> img, Image<Bgr, byte> imgCopy)
        {
            unsafe
            {
                // direct access to the image memory(sequencial)
                // direcion top left -> bottom right

                MIplImage m = img.MIplImage;
                byte* dataPtr = (byte*)m.ImageData.ToPointer(); // Pointer to the image

                MIplImage mC = imgCopy.MIplImage;
                byte* dataPtrC = (byte*)mC.ImageData.ToPointer(); // Pointer to the image

                int width = img.Width;
                int height = img.Height;
                int nChan = m.NChannels; // number of channels - 3
                int widthStep = m.WidthStep;
                int padding = widthStep - nChan * m.Width; // alinhament bytes (padding)

                //pixel do canto superior esquerdo
                if (dataPtrC[0] == 255 && dataPtrC[1] == 255 && dataPtrC[2] == 255)
                {

                    (dataPtr + nChan)[0] = 255;
                    (dataPtr + nChan)[1] = 255;
                    (dataPtr + nChan)[2] = 255;

                    (dataPtr + widthStep)[0] = 255;
                    (dataPtr + widthStep)[1] = 255;
                    (dataPtr + widthStep)[2] = 255;
                }

                dataPtr += nChan;
                dataPtrC += nChan;

                //pixeis da linha superior exceto os cantos
                for (int x = 1; x < width - 1; x++)
                {
                    if (dataPtrC[0] == 255 && dataPtrC[1] == 255 && dataPtrC[2] == 255)
                    {
                        (dataPtr - nChan)[0] = 255;
                        (dataPtr - nChan)[1] = 255;
                        (dataPtr - nChan)[2] = 255;

                        (dataPtr + nChan)[0] = 255;
                        (dataPtr + nChan)[1] = 255;
                        (dataPtr + nChan)[2] = 255;

                        (dataPtr + widthStep)[0] = 255;
                        (dataPtr + widthStep)[1] = 255;
                        (dataPtr + widthStep)[2] = 255;
                    }

                    dataPtr += nChan;
                    dataPtrC += nChan;
                }


                //pixel do canto superior direito
                if (dataPtrC[0] == 255 && dataPtrC[1] == 255 && dataPtrC[2] == 255)
                {
                    (dataPtr - nChan)[0] = 255;
                    (dataPtr - nChan)[1] = 255;
                    (dataPtr - nChan)[2] = 255;

                    (dataPtr + widthStep)[0] = 255;
                    (dataPtr + widthStep)[1] = 255;
                    (dataPtr + widthStep)[2] = 255;
                }

                dataPtr += (padding + nChan);
                dataPtrC += (padding + nChan);

                for (int y = 1; y < height - 1; y++)
                {
                    //pixeis da coluna da esquerda exceto os cantos
                    if (dataPtrC[0] == 255 && dataPtrC[1] == 255 && dataPtrC[2] == 255)
                    {
                        (dataPtr - widthStep)[0] = 255;
                        (dataPtr - widthStep)[1] = 255;
                        (dataPtr - widthStep)[2] = 255;

                        (dataPtr + nChan)[0] = 255;
                        (dataPtr + nChan)[1] = 255;
                        (dataPtr + nChan)[2] = 255;

                        (dataPtr + widthStep)[0] = 255;
                        (dataPtr + widthStep)[1] = 255;
                        (dataPtr + widthStep)[2] = 255;
                    }

                    dataPtr += nChan;
                    dataPtrC += nChan;

                    //core da imagem
                    for (int x = 1; x < width - 1; x++)
                    {
                        if (dataPtrC[0] == 255 && dataPtrC[1] == 255 && dataPtrC[2] == 255)
                        {
                            (dataPtr - widthStep)[0] = 255;
                            (dataPtr - widthStep)[1] = 255;
                            (dataPtr - widthStep)[2] = 255;

                            (dataPtr - nChan)[0] = 255;
                            (dataPtr - nChan)[1] = 255;
                            (dataPtr - nChan)[2] = 255;

                            (dataPtr + nChan)[0] = 255;
                            (dataPtr + nChan)[1] = 255;
                            (dataPtr + nChan)[2] = 255;

                            (dataPtr + widthStep)[0] = 255;
                            (dataPtr + widthStep)[1] = 255;
                            (dataPtr + widthStep)[2] = 255;
                        }

                        dataPtr += nChan;
                        dataPtrC += nChan;
                    }

                    //pixeis da coluna da direita exceto os cantos
                    if (dataPtrC[0] == 255 && dataPtrC[1] == 255 && dataPtrC[2] == 255)
                    {
                        (dataPtr - widthStep)[0] = 255;
                        (dataPtr - widthStep)[1] = 255;
                        (dataPtr - widthStep)[2] = 255;

                        (dataPtr - nChan)[0] = 255;
                        (dataPtr - nChan)[1] = 255;
                        (dataPtr - nChan)[2] = 255;

                        (dataPtr + widthStep)[0] = 255;
                        (dataPtr + widthStep)[1] = 255;
                        (dataPtr + widthStep)[2] = 255;
                    }

                    dataPtr += (padding + nChan);
                    dataPtrC += (padding + nChan);
                }

                //pixel do canto inferior esquerdo
                if (dataPtrC[0] == 255 && dataPtrC[1] == 255 && dataPtrC[2] == 255)
                {
                    (dataPtr - widthStep)[0] = 255;
                    (dataPtr - widthStep)[1] = 255;
                    (dataPtr - widthStep)[2] = 255;

                    (dataPtr + nChan)[0] = 255;
                    (dataPtr + nChan)[1] = 255;
                    (dataPtr + nChan)[2] = 255;
                }

                dataPtr += nChan;
                dataPtrC += nChan;

                //pixeis da linha inferior exceto os cantos
                for (int x = 1; x < width - 1; x++)
                {
                    if (dataPtrC[0] == 255 && dataPtrC[1] == 255 && dataPtrC[2] == 255)
                    {
                        (dataPtr - widthStep)[0] = 255;
                        (dataPtr - widthStep)[1] = 255;
                        (dataPtr - widthStep)[2] = 255;

                        (dataPtr - nChan)[0] = 255;
                        (dataPtr - nChan)[1] = 255;
                        (dataPtr - nChan)[2] = 255;

                        (dataPtr + nChan)[0] = 255;
                        (dataPtr + nChan)[1] = 255;
                        (dataPtr + nChan)[2] = 255;
                    }

                    dataPtr += nChan;
                    dataPtrC += nChan;
                }

                //pixel do canto inferior direito
                if (dataPtrC[0] == 255 && dataPtrC[1] == 255 && dataPtrC[2] == 255)
                {
                    (dataPtr - widthStep)[0] = 255;
                    (dataPtr - widthStep)[1] = 255;
                    (dataPtr - widthStep)[2] = 255;

                    (dataPtr - nChan)[0] = 255;
                    (dataPtr - nChan)[1] = 255;
                    (dataPtr - nChan)[2] = 255;
                }
            }
        }

        public static void Erosion(Image<Bgr, byte> img, Image<Bgr, byte> imgCopy)
        {
            unsafe
            {
                // direct access to the image memory(sequencial)
                // direcion top left -> bottom right

                MIplImage m = img.MIplImage;
                byte* dataPtr = (byte*)m.ImageData.ToPointer(); // Pointer to the image

                MIplImage mC = imgCopy.MIplImage;
                byte* dataPtrC = (byte*)mC.ImageData.ToPointer(); // Pointer to the image

                int width = img.Width;
                int height = img.Height;
                int nChan = m.NChannels; // number of channels - 3
                int widthStep = m.WidthStep;
                int padding = widthStep - nChan * m.Width; // alinhament bytes (padding)

                //pixel do canto superior esquerdo
                if (!(dataPtrC[0] == 255 && dataPtrC[1] == 255 && dataPtrC[2] == 255 &&
                     (dataPtrC + nChan)[0] == 255 && (dataPtrC + nChan)[1] == 255 && (dataPtrC + nChan)[2] == 255 &&
                     (dataPtrC + widthStep)[0] == 255 && (dataPtrC + widthStep)[1] == 255 && (dataPtrC + widthStep)[2] == 255))
                {
                    dataPtr[0] = 0;
                    dataPtr[1] = 0;
                    dataPtr[2] = 0;
                }

                dataPtr += nChan;
                dataPtrC += nChan;

                //pixeis da linha superior exceto os cantos
                for (int x = 1; x < width - 1; x++)
                {
                    if (!((dataPtrC - nChan)[0] == 255 && (dataPtrC - nChan)[1] == 255 && (dataPtrC - nChan)[2] == 255 &&
                           dataPtrC[0] == 255 && dataPtrC[1] == 255 && dataPtrC[2] == 255 &&
                          (dataPtrC + nChan)[0] == 255 && (dataPtrC + nChan)[1] == 255 && (dataPtrC + nChan)[2] == 255 &&
                          (dataPtrC + widthStep)[0] == 255 && (dataPtrC + widthStep)[1] == 255 && (dataPtrC + widthStep)[2] == 255))
                    {
                        (dataPtr)[0] = 0;
                        (dataPtr)[1] = 0;
                        (dataPtr)[2] = 0;
                    }

                    dataPtr += nChan;
                    dataPtrC += nChan;
                }


                //pixel do canto superior direito
                if (!((dataPtrC - nChan)[0] == 255 && (dataPtrC - nChan)[1] == 255 && (dataPtrC - nChan)[2] == 255 &&
                       dataPtrC[0] == 255 && dataPtrC[1] == 255 && dataPtrC[2] == 255 &&
                      (dataPtrC + widthStep)[0] == 255 && (dataPtrC + widthStep)[1] == 255 && (dataPtrC + widthStep)[2] == 255))
                {
                    dataPtr[0] = 0;
                    dataPtr[1] = 0;
                    dataPtr[2] = 0;
                }

                dataPtr += (padding + nChan);
                dataPtrC += (padding + nChan);

                for (int y = 1; y < height - 1; y++)
                {
                    //pixeis da coluna da esquerda exceto os cantos
                    if (!((dataPtrC - widthStep)[0] == 255 && (dataPtrC - widthStep)[1] == 255 && (dataPtrC - widthStep)[2] == 255 &&
                           dataPtrC[0] == 255 && dataPtrC[1] == 255 && dataPtrC[2] == 255 &&
                          (dataPtrC + nChan)[0] == 255 && (dataPtrC + nChan)[1] == 255 && (dataPtrC + nChan)[2] == 255 &&
                          (dataPtrC + widthStep)[0] == 255 && (dataPtrC + widthStep)[1] == 255 && (dataPtrC + widthStep)[2] == 255))
                    {
                        (dataPtr)[0] = 0;
                        (dataPtr)[1] = 0;
                        (dataPtr)[2] = 0;
                    }

                    dataPtr += nChan;
                    dataPtrC += nChan;

                    //core da imagem
                    for (int x = 1; x < width - 1; x++)
                    {
                        if (!((dataPtrC - widthStep)[0] == 255 && (dataPtrC - widthStep)[1] == 255 && (dataPtrC - widthStep)[2] == 255 &&
                              (dataPtrC - nChan)[0] == 255 && (dataPtrC - nChan)[1] == 255 && (dataPtrC - nChan)[2] == 255 &&
                               dataPtrC[0] == 255 && dataPtrC[1] == 255 && dataPtrC[2] == 255 &&
                              (dataPtrC + nChan)[0] == 255 && (dataPtrC + nChan)[1] == 255 && (dataPtrC + nChan)[2] == 255 &&
                              (dataPtrC + widthStep)[0] == 255 && (dataPtrC + widthStep)[1] == 255 && (dataPtrC + widthStep)[2] == 255))
                        {
                            (dataPtr)[0] = 0;
                            (dataPtr)[1] = 0;
                            (dataPtr)[2] = 0;
                        }

                        dataPtr += nChan;
                        dataPtrC += nChan;
                    }

                    //pixeis da coluna da direita exceto os cantos
                    if (!((dataPtrC - widthStep)[0] == 255 && (dataPtrC - widthStep)[1] == 255 && (dataPtrC - widthStep)[2] == 255 &&
                          (dataPtrC - nChan)[0] == 255 && (dataPtrC - nChan)[1] == 255 && (dataPtrC - nChan)[2] == 255 &&
                           dataPtrC[0] == 255 && dataPtrC[1] == 255 && dataPtrC[2] == 255 &&
                          (dataPtrC + widthStep)[0] == 255 && (dataPtrC + widthStep)[1] == 255 && (dataPtrC + widthStep)[2] == 255))
                    {
                        (dataPtr)[0] = 0;
                        (dataPtr)[1] = 0;
                        (dataPtr)[2] = 0;
                    }

                    dataPtr += (padding + nChan);
                    dataPtrC += (padding + nChan);
                }

                //pixel do canto inferior esquerdo
                if (!((dataPtrC - widthStep)[0] == 255 && (dataPtrC - widthStep)[1] == 255 && (dataPtrC - widthStep)[2] == 255 &&
                       dataPtrC[0] == 255 && dataPtrC[1] == 255 && dataPtrC[2] == 255 &&
                      (dataPtrC + nChan)[0] == 255 && (dataPtrC + nChan)[1] == 255 && (dataPtrC + nChan)[2] == 255))
                {
                    (dataPtr)[0] = 0;
                    (dataPtr)[1] = 0;
                    (dataPtr)[2] = 0;
                }

                dataPtr += nChan;
                dataPtrC += nChan;

                //pixeis da linha inferior exceto os cantos
                for (int x = 1; x < width - 1; x++)
                {
                    if (!((dataPtrC - widthStep)[0] == 255 && (dataPtrC - widthStep)[1] == 255 && (dataPtrC - widthStep)[2] == 255 &&
                          (dataPtrC - nChan)[0] == 255 && (dataPtrC - nChan)[1] == 255 && (dataPtrC - nChan)[2] == 255 &&
                           dataPtrC[0] == 255 && dataPtrC[1] == 255 && dataPtrC[2] == 255 &&
                          (dataPtrC + nChan)[0] == 255 && (dataPtrC + nChan)[1] == 255 && (dataPtrC + nChan)[2] == 255))
                    {
                        (dataPtr)[0] = 0;
                        (dataPtr)[1] = 0;
                        (dataPtr)[2] = 0;
                    }

                    dataPtr += nChan;
                    dataPtrC += nChan;
                }

                //pixel do canto inferior direito
                if (!((dataPtrC - widthStep)[0] == 255 && (dataPtrC - widthStep)[1] == 255 && (dataPtrC - widthStep)[2] == 255 &&
                              (dataPtrC - nChan)[0] == 255 && (dataPtrC - nChan)[1] == 255 && (dataPtrC - nChan)[2] == 255 &&
                               dataPtrC[0] == 255 && dataPtrC[1] == 255 && dataPtrC[2] == 255))
                {
                    (dataPtr)[0] = 0;
                    (dataPtr)[1] = 0;
                    (dataPtr)[2] = 0;
                }
            }
        }

        public static int[,] ConnectedComponents(Image<Bgr, byte> img, int level)
        {
            unsafe
            {
                MIplImage mDestino = img.MIplImage;
                byte* dataPtrDestino = (byte*)mDestino.ImageData.ToPointer();

                int h = mDestino.Height;
                int w = mDestino.Width;
                int widthStep = mDestino.WidthStep;
                int x = 0, y = 0;
                int nC = mDestino.NChannels; // number of channels = 3
                //numero para identificar as etiquetas
                int numero_etiqueta = 1;

                //matrix das etiquetas
                int[,] matriz = new int[h, w];
                int black;

                black = (dataPtrDestino + y * widthStep + x * nC)[0];

                if (y == 0 && x == 0 && black == 0)
                {
                    matriz[0, 0] = numero_etiqueta++;
                }
                //processar a primeira linha que so tem valor a cima
                for (x = 1; x < w; x++)
                {
                    black = (dataPtrDestino + y * widthStep + x * nC)[0];

                    if (black == 0)
                    {
                        if (matriz[0, x - 1] == 0)
                        {
                            matriz[0, x] = matriz[0, x - 1];
                        }
                        else
                        {
                            matriz[0, x] = numero_etiqueta++;
                        }
                    }
                    else if (black == 255)
                    {
                        matriz[0, x] = 0;
                    }
                }

                //processar a primeira coluna que so tem valor a cima
                for (y = 1; y < h; y++)
                {
                    black = (dataPtrDestino + y * widthStep + x * nC)[0];

                    if (black == 0)
                    {
                        if (matriz[y - 1, 0] == 0)
                        {
                            matriz[y, 0] = matriz[y - 1, 0];
                        }
                        else
                        {
                            matriz[y, 0] = numero_etiqueta++;
                        }
                    }
                    else if (black == 255)
                    {
                        matriz[y, 0] = 0;
                    }
                }

                //Dá as etiquetas e processa o core - top/left 

                for (y = 1; y < h; y++)
                {
                    for (x = 1; x < w; x++)
                    {
                        black = (dataPtrDestino + y * widthStep + x * nC)[0];

                        if (black == 0)
                        {
                            if (matriz[y - 1, x] != 0 && matriz[y, x - 1] != 0)
                            {
                                if (matriz[y - 1, x] < matriz[y, x - 1])
                                {
                                    matriz[y, x] = matriz[y - 1, x];
                                }
                                else
                                {
                                    matriz[y, x] = matriz[y, x - 1];
                                }

                            }

                            else if (matriz[y - 1, x] == 0 && matriz[y, x - 1] != 0)
                            {
                                matriz[y, x] = matriz[y, x - 1];
                            }

                            else if (matriz[y - 1, x] != 0 && matriz[y, x - 1] == 0)
                            {
                                matriz[y, x] = matriz[y - 1, x];
                            }
                            else
                            {
                                matriz[y, x] = numero_etiqueta++;
                            }
                        }
                        else if (black == 255)
                        {
                            matriz[y, x] = 0;
                        }
                    }
                }

                if (level == 2 || level == 4)
                {

                    ////processa o core - top/right
                    for (y = 1; y < h; y++)

                    {

                        for (x = w - 2; x >= 0; x--)
                        {
                            if (matriz[y, x] != 0)
                            {
                                if (matriz[y, x + 1] != 0 && matriz[y, x + 1] < matriz[y, x])
                                    matriz[y, x] = matriz[y, x + 1];
                                else if (matriz[y - 1, x] != 0 && matriz[y - 1, x] < matriz[y, x])
                                    matriz[y, x] = matriz[y - 1, x];
                            }
                        }
                    }

                    //processa o core - bottom/right
                    for (y = h - 2; y >= 0; y--)

                    {

                        for (x = w - 2; x >= 0; x--)
                        {
                            if (matriz[y, x] != 0)
                            {
                                if (matriz[y, x] != 0)
                                {
                                    if (matriz[y, x + 1] != 0 && matriz[y, x + 1] < matriz[y, x])
                                        matriz[y, x] = matriz[y, x + 1];
                                    else if (matriz[y + 1, x] != 0 && matriz[y + 1, x] < matriz[y, x])
                                        matriz[y, x] = matriz[y + 1, x];
                                }
                            }
                        }
                    }
                }

                return matriz;

            }
        }

        public static string DirectBinary(Image<Bgr, byte> img, Image<Bgr, byte> imgCopy)
        {
            unsafe
            {
                MIplImage origem = img.MIplImage;
                byte* dataPtr_o = (byte*)origem.ImageData.ToPointer(); // Pointer to the image

                int width = origem.Width;
                int height = origem.Height;
                int widthStep = origem.WidthStep;
                int nChan = origem.NChannels; // number of channels = 3
                int black;
                int[] lista = new int[256];
                string msg = "";

                //dividir a imagem em 20 partes
                int h = Convert.ToInt32(height / 20);
                int w = Convert.ToInt32(width / 20);

                int x = 0;
                int y = 0;
                //parte superior 
                for (y = 0; y < 8; y++)
                {
                    for (x = 8; x < 13; x++)
                    {
                        black = (dataPtr_o + h / 2 + y * widthStep * h + w / 2 + x * nChan * w)[0];
                        if (black == 255)
                        {
                            black = 0;
                        }
                        else
                        {
                            black = 1;
                        }
                        msg = msg + black.ToString();

                    }
                }

                //parte do meio 
                for (y = 8; y < 13; y++)
                {
                    for (x = 0; x < 21; x++)
                    {
                        black = (dataPtr_o + h / 2 + y * widthStep * h + w / 2 + x * nChan * w)[0];
                        if (black == 255)
                        {
                            black = 0;
                        }
                        else
                        {
                            black = 1;
                        }
                        msg = msg + black.ToString();
                    }
                }

                //parte inferior
                for (y = 13; y < 21; y++)
                {
                    for (x = 8; x < 21; x++)
                    {
                        black = (dataPtr_o + h / 2 + y * widthStep * h + w / 2 + x * nChan * w)[0];
                        if (black == 255)
                        {
                            black = 0;
                        }
                        else
                        {
                            black = 1;
                        }
                        msg = msg + black.ToString();

                    }
                }

                return msg;
            }

        }

        public static void RotationQR(Image<Bgr, byte> imgCopy, Image<Bgr, byte> imgOrigem, float angle, int centerx, int centery)
        {
            unsafe
            {
                // direct access to the image memory(sequencial)
                // direcion top left -> bottom right

                MIplImage origem = imgOrigem.MIplImage;
                byte* dataPtr_o = (byte*)origem.ImageData.ToPointer(); // Pointer to the image

                MIplImage copy = imgCopy.MIplImage;
                byte* dataPtr_c = (byte*)copy.ImageData.ToPointer(); // Pointer to the image

                int width = origem.Width;
                int height = origem.Height;
                int nChan = origem.NChannels; // number of channels - 3
                int padding = origem.WidthStep - origem.NChannels * origem.Width; // alinhament bytes (padding)
                int x, y, xo, yo;
                int widthstep = origem.WidthStep;
                byte blue, green, red;

                double H = centery;
                double W = centerx;

                angle = -(float)((angle / 180) * Math.PI);

                double cos = Math.Cos(angle);
                double sin = Math.Sin(angle);

                if (nChan == 3) // image in RGB
                {
                    for (y = 0; y < height; y++)
                    {
                        for (x = 0; x < width; x++)
                        {
                            //retrieve 3 color componets from copy
                            yo = (int)(Math.Round(H - (x - W) * sin - (H - y) * cos));
                            xo = (int)(Math.Round((x - W) * cos - (H - y) * sin + W));


                            if ((xo < width && xo >= 0) && (yo < height && yo >= 0))
                            {
                                //retrieve 3 colour components from origem
                                blue = (byte)(dataPtr_o + yo * widthstep + xo * nChan)[0];
                                green = (byte)(dataPtr_o + yo * widthstep + xo * nChan)[1];
                                red = (byte)(dataPtr_o + yo * widthstep + xo * nChan)[2];

                                dataPtr_c[0] = blue;
                                dataPtr_c[1] = green;
                                dataPtr_c[2] = red;

                            }
                            else
                            {
                                dataPtr_c[0] = 255;
                                dataPtr_c[1] = 255;
                                dataPtr_c[2] = 255;
                            }
                            // advance the pointer to the next pixel
                            dataPtr_c += nChan;
                        }

                        //at the end of the line advance the pointer by the aligment bytes (padding)
                        dataPtr_c += padding;
                    }
                }                
            }
        }

        public static IDictionary<int, (int, int)> Centroides(int[,] matriz)
        {
            unsafe
            {
                int y = 0;
                int x = 0;

                IDictionary<int, (int, int, int)> areacentro = new Dictionary<int, (int, int, int)>(); //área x e y do centroide

                for (x = 0; x < matriz.GetLength(0); x++)
                {
                    for (y = 0; y < matriz.GetLength(1); y++)
                    {
                        if (matriz[x, y] != 0)
                        {
                            if (areacentro.ContainsKey(matriz[x, y]))
                            {

                                (int a, int c, int b) = areacentro[matriz[x, y]];
                                a++;
                                b = b + (x + 1);
                                c = c + (y + 1);

                                areacentro[matriz[x, y]] = (a, c, b);

                            }
                            else
                            {
                                int b = (x + 1);
                                int c = (y + 1);

                                areacentro.Add(matriz[x, y], (1, c, b));
                            }
                        }
                    }
                }

                foreach (var objeto in areacentro.ToList()) //Obtem os centroides de cada objeto
                    areacentro[objeto.Key] = (objeto.Value.Item1, objeto.Value.Item2 / objeto.Value.Item1, objeto.Value.Item3 / objeto.Value.Item1);

                IDictionary<int, (int, int)> quadrados = new Dictionary<int, (int, int)>(); //x e y do centroide

                int quad = 1;



                foreach (var objeto1 in areacentro)
                {

                    foreach (var objeto2 in areacentro)
                    {
                        int difx = (objeto1.Value.Item2 - objeto2.Value.Item2);
                        int dify = (objeto1.Value.Item3 - objeto2.Value.Item3);

                        if (difx == 0 && dify == 0 && objeto1.Key != objeto2.Key)
                        {
                            if (objeto1.Value.Item1 / objeto2.Value.Item1 == 2)
                            {
                                quadrados.Add(quad, (objeto1.Value.Item3, objeto1.Value.Item2));
                                quad++;
                            }

                        }

                    }
                }

                foreach (var objeto1 in areacentro)
                {
                    foreach (var objeto2 in areacentro)
                    {
                        int difx = (objeto1.Value.Item2 - objeto2.Value.Item2);
                        int dify = (objeto1.Value.Item3 - objeto2.Value.Item3);

                        if (difx >= -1 && difx <= 1 && dify >= -1 && dify <= 1 && (difx != 0 || dify != 0) && objeto1.Key != objeto2.Key && quadrados.Count < 3)
                        {
                            if (objeto1.Value.Item1 / objeto2.Value.Item1 == 2)
                            {
                                quadrados.Add(quad, (objeto1.Value.Item3, objeto1.Value.Item2));
                                quad++;
                            }
                        }
                    }
                }
                return quadrados;
            }
        }

        public static (int, int) CentroCentroides(IDictionary<int, (int, int)> quadrados)
        {
            unsafe
            {
                (int y_quad1, int x_quad1) quadrado1 = quadrados[1];
                (int y_quad2, int x_quad2) quadrado2 = quadrados[2];
                (int y_quad3, int x_quad3) quadrado3 = quadrados[3];

                //ver distancia maior entre centroides 
                int distance12 = (int)Math.Sqrt(Math.Pow(quadrado1.y_quad1 - quadrado2.y_quad2, 2) + Math.Pow(quadrado1.x_quad1 - quadrado2.x_quad2, 2));
                int distance13 = (int)Math.Sqrt(Math.Pow(quadrado1.y_quad1 - quadrado3.y_quad3, 2) + Math.Pow(quadrado1.x_quad1 - quadrado3.x_quad3, 2));
                int distance23 = (int)Math.Sqrt(Math.Pow(quadrado2.y_quad2 - quadrado3.y_quad3, 2) + Math.Pow(quadrado2.x_quad2 - quadrado3.x_quad3, 2));

                int max_dis = distance12;
                int centerx = (quadrado1.x_quad1 + quadrado2.x_quad2) / 2;
                int centery = (quadrado1.y_quad1 + quadrado2.y_quad2) / 2;

                if (max_dis < distance13)
                {
                    max_dis = distance13;
                    centerx = (quadrado1.x_quad1 + quadrado3.x_quad3) / 2;
                    centery = (quadrado1.y_quad1 + quadrado3.y_quad3) / 2;
                }

                if (max_dis < distance23)
                {
                    centerx = (quadrado2.x_quad2 + quadrado3.x_quad3) / 2;
                    centery = (quadrado2.y_quad2 + quadrado3.y_quad3) / 2;
                }
                return (centerx, centery);
            }
        }

        public static float RotacaoCentroides(IDictionary<int, (int, int)> quadrados, int[,] matriz)
        {
            unsafe
            {
                (int, int) ponto_sup = (0, 0);
                (int, int) ponto_inf = (0, 0);
                int found = 0;
                int x = 0;
                int y = 0;

                //Ponto superior
                for (y = 0, found = 0; y < matriz.GetLength(0); y++)
                {

                    for (x = matriz.GetLength(1) - 1; x > 0; x--)
                    {

                        if (matriz[y, x] != 0)
                        {
                            ponto_sup = (y, x);
                            found = 1;
                            break;
                        }

                    }
                    if (found == 1)
                        break;
                }

                //ponto inferior
                for (y = matriz.GetLength(0) - 1, found = 0; y > 0; y--)
                {
                    for (x = matriz.GetLength(1) - 1; x > 0; x--)
                    {

                        if (matriz[y, x] != 0)
                        {
                            ponto_inf = (y, x);
                            found = 1;
                            break;
                        }

                    }
                    if (found == 1)
                        break;
                }



                (int y_quad1, int x_quad1) quadrado1 = quadrados[1];
                (int y_quad2, int x_quad2) quadrado2 = quadrados[2];
                (int y_quad3, int x_quad3) quadrado3 = quadrados[3];
                float rotation = 0;

                //Obtem a distancia do ponto superior aos 3 quadrados
                double sup_distance1 = Math.Sqrt(Math.Pow(ponto_sup.Item1 - quadrado1.y_quad1, 2) + Math.Pow(ponto_sup.Item2 - quadrado1.x_quad1, 2));
                double sup_distance2 = Math.Sqrt(Math.Pow(ponto_sup.Item1 - quadrado2.y_quad2, 2) + Math.Pow(ponto_sup.Item2 - quadrado2.x_quad2, 2));
                double sup_distance3 = Math.Sqrt(Math.Pow(ponto_sup.Item1 - quadrado3.y_quad3, 2) + Math.Pow(ponto_sup.Item2 - quadrado3.x_quad3, 2));

                //Verifica de qual quadrado está mais próximo
                List<double> list_sup = new List<double>() { sup_distance1, sup_distance2, sup_distance3 };
                double sup_min = list_sup.Min();

                //Obtem a distancia do ponto inferior aos 3 quadrados
                double inf_distance1 = Math.Sqrt(Math.Pow(ponto_inf.Item1 - quadrado1.y_quad1, 2) + Math.Pow(ponto_inf.Item2 - quadrado1.x_quad1, 2));
                double inf_distance2 = Math.Sqrt(Math.Pow(ponto_inf.Item1 - quadrado2.y_quad2, 2) + Math.Pow(ponto_inf.Item2 - quadrado2.x_quad2, 2));
                double inf_distance3 = Math.Sqrt(Math.Pow(ponto_inf.Item1 - quadrado3.y_quad3, 2) + Math.Pow(ponto_inf.Item2 - quadrado3.x_quad3, 2));

                //Verifica de qual quadrado está mais próximo
                List<double> list_inf = new List<double>() { inf_distance1, inf_distance2, inf_distance3 };
                double inf_min = list_inf.Min();


                if (inf_min < sup_min)
                {
                    if (inf_min == inf_distance1)
                        rotation = (float)Math.Atan2(ponto_inf.Item1 - quadrado1.y_quad1, quadrado1.x_quad1 - ponto_inf.Item2);
                    if (inf_min == inf_distance2)
                        rotation = (float)Math.Atan2(ponto_inf.Item1 - quadrado2.y_quad2, quadrado2.x_quad2 - ponto_inf.Item2);
                    if (inf_min == inf_distance3)
                        rotation = (float)Math.Atan2(ponto_inf.Item1 - quadrado3.y_quad3, quadrado3.x_quad3 - ponto_inf.Item2);
                }
                else
                {
                    if (sup_min == sup_distance1)
                        rotation = (float)Math.Atan2(quadrado1.y_quad1 - ponto_sup.Item1, quadrado1.x_quad1 - ponto_sup.Item2);
                    if (sup_min == sup_distance2)
                        rotation = (float)Math.Atan2(quadrado2.y_quad2 - ponto_sup.Item1, quadrado2.x_quad2 - ponto_sup.Item2);
                    if (sup_min == sup_distance3)
                        rotation = (float)Math.Atan2(quadrado3.y_quad3 - ponto_sup.Item1, quadrado3.x_quad3 - ponto_sup.Item2);
                }

                float r2 = rotation * (180 / (float)Math.PI);

                if (r2 > 0 && r2 <= 180)
                    r2 = r2 - 45;

                // checkar qual deveria ser o sinal
                if (ponto_sup.Item2 > ponto_inf.Item2)
                {
                    r2 = -r2;
                }
                return r2;
            }
        }

        public static (int, int, int, int, int, int, float) QRPlaces(Image<Bgr, byte> img, Image<Bgr, byte> imgCopy, IDictionary<int, (int, int)> quadrados, int centrox, int centroy, float r2)
        {
            unsafe
            {
                int y_maior = 0;
                int x_maior = 0;
                int y_menor = 0;
                int x_menor = 0;

                int count_maior_y = 0;
                int count_maior_x = 0;

                int UL_x = 0;
                int UL_y = 0;
                int UR_x = 0;
                int UR_y = 0;
                int LL_x = 0;
                int LL_y = 0;

                //confirma se quadrados estão no sitio certo
                foreach (var objeto1 in quadrados)
                {
                    if (objeto1.Value.Item1 >= centroy)
                    {
                        count_maior_y++;
                        y_maior = objeto1.Value.Item1;
                    }
                    if (objeto1.Value.Item2 >= centrox)
                    {
                        count_maior_x++;
                        x_maior = objeto1.Value.Item2;
                    }
                    if (objeto1.Value.Item1 < centroy)
                        y_menor = objeto1.Value.Item1;

                    if (objeto1.Value.Item2 < centrox)
                        x_menor = objeto1.Value.Item2;
                }

                //confirma se é preciso mais rotacao
                if (count_maior_y > 1 && r2 != 45)// precisamos de rodar mais 180º
                {
                    r2 = r2 + 90;
                    RotationQR(img, imgCopy, 90, centrox, centroy);
                }

                if (r2 == 90)
                {
                    UL_x = x_maior;
                    UL_y = y_menor;

                    LL_x = x_menor;
                    LL_y = y_menor;

                    UR_x = x_maior;
                    UR_y = y_maior;
                }
                else if (r2 == 180)
                {
                    UL_x = x_maior;
                    UL_y = y_maior;

                    LL_x = x_maior;
                    LL_y = y_menor;

                    UR_x = x_menor;
                    UR_y = y_maior;
                }
                else
                {
                    foreach (var objeto1 in quadrados)
                    {
                        if (objeto1.Value.Item1 <= centroy && objeto1.Value.Item2 <= centrox) //esquerda superior
                        {
                            UL_x = objeto1.Value.Item2;
                            UL_y = objeto1.Value.Item1;
                        }
                        else if (objeto1.Value.Item1 <= centroy && objeto1.Value.Item2 >= centrox) //direita superior
                        {
                            UR_x = objeto1.Value.Item2;
                            UR_y = objeto1.Value.Item1;
                        }
                        else if (objeto1.Value.Item1 >= centroy && objeto1.Value.Item2 <= centrox) //esquerda inferior
                        {
                            LL_x = objeto1.Value.Item2;
                            LL_y = objeto1.Value.Item1;
                        }
                    }

                    if (r2 == 45)
                    {
                        LL_x = UL_x;
                        LL_y = UL_y;

                        UL_x = LL_y;
                        UL_y = LL_x;
                    }
                }

                return (UL_x, UL_y, UR_x, UR_y, LL_x, LL_y, r2);
            }

        }

        public static void CleanBG(Image<Bgr, byte> img)
        {
            unsafe
            {
                MIplImage origem = img.MIplImage;
                byte* dataPtr_o = (byte*)origem.ImageData.ToPointer(); // Pointer to the image

                int width = origem.Width;
                int height = origem.Height;
                int nChan = origem.NChannels; // number of channels - 3
                int widthstep = origem.WidthStep; // alinhament bytes (padding)
                int blue, green, red, gray;

                int count = 0;
                int count1 = 0;

                int y, x = 0;

                IDictionary<int, int> linha_a_apagar = new Dictionary<int, int>();
                IDictionary<int, int> coluna_a_apagar = new Dictionary<int, int>();

                //remove cores da imagem
                for (y = 0; y < height; y++)
                {

                    for (x = 0; x < width; x++)
                    {
                        blue = (dataPtr_o + y * widthstep + x * nChan)[0];
                        green = (dataPtr_o + y * widthstep + x * nChan)[1];
                        red = (dataPtr_o + y * widthstep + x * nChan)[2];

                        // convert to gray
                        gray = (int)Math.Round((blue + green + red) / 3.0);

                        if (gray < 5)
                        {
                            // store in the image
                            (dataPtr_o + y * widthstep + x * nChan)[0] = 0;
                            (dataPtr_o + y * widthstep + x * nChan)[1] = 0;
                            (dataPtr_o + y * widthstep + x * nChan)[2] = 0;
                        }
                        else
                        {
                            (dataPtr_o + y * widthstep + x * nChan)[0] = 255;
                            (dataPtr_o + y * widthstep + x * nChan)[1] = 255;
                            (dataPtr_o + y * widthstep + x * nChan)[2] = 255;
                        }
                    }
                }

                //identifica colunas com demasiado preto 
                for (x = 0; x < width; x++)
                {
                    int count_black = 0;

                    for (y = 0; y < height; y++)
                    {
                        gray = (dataPtr_o + y * widthstep + x * nChan)[0];

                        if (gray != 255) // se for preto acrescenta 1
                            count_black++;

                        if (count_black > (height * 0.7))
                        {

                            coluna_a_apagar.Add(count1, x);
                            count1++;
                        }
                    }
                }

                //identifica linhas com demasiado preto 
                for (y = 0; y < height; y++)
                {
                    int count_black = 0;

                    for (x = 0; x < width; x++)
                    {
                        gray = (dataPtr_o + y * widthstep + x * nChan)[0];

                        if (gray != 255) // se for preto acrescenta 1
                            count_black++;

                        if (count_black > (width * 0.65))
                        {

                            linha_a_apagar.Add(count, y);
                            count++;
                        }
                    }
                }

                //remove colunas com demasiado preto
                foreach (var objeto in coluna_a_apagar)
                {
                    int coluna = objeto.Value;
                    for (y = 0; y < height; y++)
                    {
                        (dataPtr_o + y * widthstep + coluna * nChan)[0] = 255;
                        (dataPtr_o + y * widthstep + coluna * nChan)[1] = 255;
                        (dataPtr_o + y * widthstep + coluna * nChan)[2] = 255;
                    }
                }

                //remove linhas com demasiado preto
                foreach (var objeto in linha_a_apagar)
                {
                    int linha = objeto.Value;
                    for (x = 0; x < width; x++)
                    {
                        (dataPtr_o + linha * widthstep + x * nChan)[0] = 255;
                        (dataPtr_o + linha * widthstep + x * nChan)[1] = 255;
                        (dataPtr_o + linha * widthstep + x * nChan)[2] = 255;
                    }
                }

            }
        }

        public static (int, int, int, int, int, int) FindQR(Image<Bgr, byte> img)
        {
            unsafe
            {
                MIplImage origem = img.MIplImage;
                byte* dataPtr_o = (byte*)origem.ImageData.ToPointer(); // Pointer to the image

                int width = origem.Width;
                int height = origem.Height;
                int nChan = origem.NChannels; // number of channels - 3
                int widthstep = origem.WidthStep; // alinhament bytes (padding)
                int canto;

                int count = 0;
                int y, x = 0;

                int y_superior = 0;
                int y_inferior = 0;
                int x_esquerdo = 0;
                int x_direito = 0;

                int linha_branca = 0;

                //identifica linhas com extremos
                for (y = 0; y < height; y++)
                {
                    int count_black = 0;
                    int count_white = 0;

                    for (x = 0; x < width; x++)
                    {
                        canto = (dataPtr_o + y * widthstep + x * nChan)[0];

                        if (canto != 255) // se for preto acrescenta 1
                            count_black++;
                        if (canto != 0)
                            count_white++;
                    }

                    if (count_black > (width * 0.10) && count_black < (width * 0.80) && count == 0)
                    {
                        y_superior = y;
                        //x_superior = x;
                        count = 1;
                    }

                    if (count_white > (width * 0.9) && count == 1)
                    {
                        linha_branca++;

                        if (linha_branca == 60)
                            break;
                    }

                    if (count_black > (width * 0.10) && count_black < (width * 0.80) && count == 1)
                    {
                        y_inferior = y;
                        //x_inferior = x;
                        linha_branca = 0;
                    }
                }

                //identifica colunas com extremos
                count = 0;

                for (x = 0; x < width; x++)
                {
                    int count_black = 0;
                    int count_white = 0;

                    for (y = 0; y < height; y++)
                    {
                        canto = (dataPtr_o + y * widthstep + x * nChan)[0];

                        if (canto != 255) // se for preto acrescenta 1
                            count_black++;
                        if (canto != 0)
                            count_white++;
                    }

                    if (count_black > (height * 0.10) && count_black < (height * 0.80) && count == 0)
                    {
                        x_esquerdo = x;
                        //y_esquerdo = y;
                        count = 1;
                    }

                    if (count_white > (height * 0.9) && count == 1)
                    {
                        linha_branca++;

                        if (linha_branca == 60)
                            break;
                    }

                    if (count_black > (height * 0.10) && count_black < (height * 0.80) && count == 1)
                    {
                        x_direito = x;
                        //y_direito = y;
                        linha_branca = 0;
                    }
                }

                int largura = x_direito - x_esquerdo;
                int altura = y_inferior - y_superior;

                Rectangle img_cortada;
                img_cortada = new Rectangle(x_esquerdo, y_superior, largura, altura);
                img.Draw(img_cortada, new Bgr(0, 0, 255), 1);


                return (y_superior, y_inferior, x_esquerdo, x_direito, largura, altura);
            }

        }

        public static void QRCodeReader(Image<Bgr, byte> img, Image<Bgr, byte> imgCopy, int level, out int Center_x, out int Center_y, out int Width, out int Height, out float Rotation, out string BinaryOut, out int UL_x_out, out int UL_y_out, out int UR_x_out, out int UR_y_out, out int LL_x_out, out int LL_y_out)
        {
            unsafe
            {
                MIplImage origem = img.MIplImage;
                byte* dataPtr_o = (byte*)origem.ImageData.ToPointer(); // Pointer to the image

                string msg = "";
                int centerx = 0;
                int centery = 0;
                float r2 = 0;
                int width = 0;
                int height = 0;

                int UL_x = 0;
                int UL_y = 0;
                int UR_x = 0;
                int UR_y = 0;
                int LL_x = 0;
                int LL_y = 0;


                if (level == 1)
                {
                    int[,] matriz = ConnectedComponents(img, level);

                    (int y_up, int y_down, int x_left, int x_right, int largura, int altura) = FindQR(img);
                    Rectangle retangulo = new Rectangle(x_left + 1, y_up + 1, largura, altura);
                    Image<Bgr, byte> QR = img.Copy(retangulo);
                    IDictionary<int, (int, int)> quadrados = new Dictionary<int, (int, int)>(); //x e y do centroide
                    quadrados = Centroides(matriz);

                    msg = DirectBinary(QR, QR.Copy());

                    (int centrox, int centroy) = CentroCentroides(quadrados);
                    (int ul_x, int ul_y, int ur_x, int ur_y, int ll_x, int ll_y, float rot) = QRPlaces(img, img.Copy(), quadrados, centrox, centroy, r2);
                    width = largura;
                    height = altura;
                    centerx = centrox;
                    centery = centroy;
                    UL_x = ul_x;
                    UL_y = ul_y;
                    UR_x = ur_x;
                    UR_y = ur_y;
                    LL_x = ll_x;
                    LL_y = ll_y;
                    r2 = rot;

                }

                else if (level == 2)
                {
                    int[,] matriz = ConnectedComponents(img, level);

                    IDictionary<int, (int, int)> quadrados = new Dictionary<int, (int, int)>(); //x e y do centroide
                    quadrados = Centroides(matriz);

                    r2 = RotacaoCentroides(quadrados, matriz);
                    (int centrox, int centroy) = CentroCentroides(quadrados);

                    RotationQR(img, img.Copy(), r2, centrox, centroy);

                    (int ul_x, int ul_y, int ur_x, int ur_y, int ll_x, int ll_y, float rot) = QRPlaces(img, img.Copy(), quadrados, centrox, centroy, r2);

                    (int y_up, int y_down, int x_left, int x_right, int largura, int altura) = FindQR(img);
                    Rectangle retangulo = new Rectangle(x_left + 1, y_up + 1, largura - 1, altura - 1);
                    Image<Bgr, byte> QR = img.Copy(retangulo);
                    msg = DirectBinary(QR, QR.Copy());

                    width = largura;
                    height = altura;

                    centerx = centrox;
                    centery = centroy;
                    UL_x = ul_x;
                    UL_y = ul_y;
                    UR_x = ur_x;
                    UR_y = ur_y;
                    LL_x = ll_x;
                    LL_y = ll_y;
                    r2 = rot;
                }

                else if (level == 3)
                {
                    Image<Bgr, Byte> imgUndo = null; // undo backup image - UNDO
                    imgUndo = img.Copy();

                    CleanBG(img);

                    int[,] matriz = ConnectedComponents(img, level);
                    (int y_up, int y_down, int x_left, int x_right, int largura, int altura) = FindQR(img);

                    IDictionary<int, (int, int)> quadrados = new Dictionary<int, (int, int)>(); //x e y do centroide
                    quadrados = Centroides(matriz);

                    (int centrox, int centroy) = CentroCentroides(quadrados);

                    (int ul_x, int ul_y, int ur_x, int ur_y, int ll_x, int ll_y, float rot) = QRPlaces(img, img.Copy(), quadrados, centrox, centroy, 0);

                    Rectangle retangulo = new Rectangle(x_left + 6, y_up + 6, largura, altura);
                    Image<Bgr, byte> QR = imgUndo.Copy(retangulo);
                    
                    ConvertToBW(QR, 250);

                    msg = DirectBinary(QR, QR.Copy());
                    centerx = centrox;
                    centery = centroy;

                    width = largura;
                    height = altura;

                    UL_x = ul_x;
                    UL_y = ul_y;
                    UR_x = ur_x;
                    UR_y = ur_y;
                    LL_x = ll_x;
                    LL_y = ll_y;
                    r2 = 0;
                }

                else if (level == 4)
                {
                    Image<Bgr, Byte> imgUndo = null; // undo backup image - UNDO
                    imgUndo = img.Copy();

                    Image<Bgr, Byte> imgQR = img.Copy(); // undo backup image - UNDO

                    int[,] matriz = ConnectedComponents(img, level);

                    IDictionary<int, (int, int)> quadrados = new Dictionary<int, (int, int)>(); //x e y do centroide
                    quadrados = Centroides(matriz);

                    //r2 = RotacaoCentroides(quadrados, matriz);
                    (int centrox, int centroy) = CentroCentroides(quadrados);

                    (int ul_x, int ul_y, int ur_x, int ur_y, int ll_x, int ll_y, float rot) = QRPlaces(img, img.Copy(), quadrados, centrox, centroy, r2);

                    Dilatation(img, img.Copy());
                    CleanBG(img);
                    Erosion(img, img.Copy());
                    Mean(img, img.Copy());


                    (int y_up, int y_down, int x_left, int x_right, int largura, int altura) = FindQR(img);

                    Rectangle retangulo = new Rectangle(x_left, y_up, largura, altura);
                    Image<Bgr, byte> QR = imgQR.Copy(retangulo);
                    msg = DirectBinary(QR, QR.Copy());

                    width = largura;
                    height = altura;

                    //ConvertToBW_Otsu(QR);

                    centerx = centrox;
                    centery = centroy;
                    UL_x = ul_x;
                    UL_y = ul_y;
                    UR_x = ur_x;
                    UR_y = ur_y;
                    LL_x = ll_x;
                    LL_y = ll_y;
                    r2 = rot;


                }

                Center_x = centerx;
                Center_y = centery;
                Width = width;
                Height = height;
                Rotation = r2;

                BinaryOut = msg;

                LL_x_out = LL_x;
                LL_y_out = LL_y;
                UL_x_out = UL_x;
                UL_y_out = UL_y;
                UR_x_out = UR_x;
                UR_y_out = UR_y;

            }
        }
    }
}

