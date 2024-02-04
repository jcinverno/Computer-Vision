import pyopencl as cl
import imageForms as iF
import numpy as np
from imageForms import *
import cv2 as cv
import tkinter as tk
from tkinter import filedialog
import numpy as np, cv2
import time

root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()


def brightness_contrast():
    global program

    try:
        # Get the platforms and devices
        platforms = cl.get_platforms()
        platform = platforms[0]
        devices = platform.get_devices()
        device = devices[0]

        # Create a context and command queue
        ctx = cl.Context([device])
        commQ = cl.CommandQueue(ctx)

        file = open("program_source.cl", "r")
        program = cl.Program(ctx, file.read())
        program.build()
        img = cv.imread(file_path)
        arrayIn = np.array(img)

        memBuffer = cl.Buffer(ctx, cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_ONLY, hostbuf=arrayIn)
        outputBuffer = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=arrayIn.nbytes)
        width = np.int32(img.shape[0])
        height = np.int32(img.shape[1])
        padding = np.int32(img.strides[0] - img.shape[0] * img.strides[1] * img.itemsize)

        brightness = np.int32(10)
        contrast = np.float32(100)

        kernelName = program.adjust_brightness
        kernelName.set_arg(0, memBuffer)
        kernelName.set_arg(1, height)
        kernelName.set_arg(2, width)
        kernelName.set_arg(3, padding)
        kernelName.set_arg(4, outputBuffer)
        kernelName.set_arg(5, brightness)
        kernelName.set_arg(6, contrast)
        size = width * height
        # Launch the kernel (use a work group size of 10)

        globalWorkSize = ((width + padding), (height + padding))
        workGroupSize = (16, 16)

        kernelEvent = cl.enqueue_nd_range_kernel(queue=commQ, kernel=kernelName, global_work_size=globalWorkSize,
                                                 local_work_size=workGroupSize)

        # Wait for the kernel to finish
        kernelEvent.wait()

        cl.enqueue_copy(commQ, arrayIn, memBuffer)

        new_img = np.empty_like(arrayIn)
        cl.enqueue_copy(commQ, new_img, outputBuffer)

        iF.showSideBySideImages(img, new_img, "", False, False)
        # Release OpenCL resources
        memBuffer.release()
        outputBuffer.release()

    except Exception as e:
        print(e)
        return False
    return True


def brightness_contrast_openCL():
    global program

    try:
        platforms = cl.get_platforms()
        platform = platforms[0]
        devices = platform.get_devices()
        device = devices[0]

        # Create a context and command queue
        ctx = cl.Context([device])
        commQ = cl.CommandQueue(ctx)

        file = open("program_source.cl", "r")
        program = cl.Program(ctx, file.read())
        program.build()
        img = cv.imread(file_path)
        arrayIn = np.array(img)

        memBuffer = cl.Buffer(ctx, cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_ONLY, hostbuf=arrayIn)
        outputBuffer = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=arrayIn.nbytes)
        width = np.int32(img.shape[0])
        height = np.int32(img.shape[1])
        padding = np.int32(img.strides[0] - img.shape[0] * img.strides[1] * img.itemsize)

        brightness = np.int32(10)
        contrast = np.float32(100)

        kernelName = program.adjust_brightness
        kernelName.set_arg(0, memBuffer)
        kernelName.set_arg(1, height)
        kernelName.set_arg(2, width)
        kernelName.set_arg(3, padding)
        kernelName.set_arg(4, outputBuffer)
        kernelName.set_arg(5, brightness)
        kernelName.set_arg(6, contrast)
        size = width * height
        # Launch the kernel (use a work group size of 10)

        globalWorkSize = ((width + padding), (height + padding))
        workGroupSize = (16, 16)

        kernelEvent = cl.enqueue_nd_range_kernel(queue=commQ, kernel=kernelName, global_work_size=globalWorkSize,
                                                 local_work_size=workGroupSize)

        # Wait for the kernel to finish
        kernelEvent.wait()

        cl.enqueue_copy(commQ, arrayIn, memBuffer)

        new_img = np.empty_like(arrayIn)
        cl.enqueue_copy(commQ, new_img, outputBuffer)

        # Release OpenCL resources
        memBuffer.release()
        outputBuffer.release()

    except Exception as e:
        print(e)
        return False
    return True


def brightness_contrast_opencv():
    image = cv.imread(file_path)
    adjusted_image = cv2.convertScaleAbs(image, alpha=10, beta=100)


def compare_times():
    start_time = time.time()
    brightness_contrast_openCL()
    print("- execute OpenCL --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    brightness_contrast_opencv()
    print("- execute OpenCV--- %s seconds ---" % (time.time() - start_time))


def sobel_threshold():
    global program

    try:
        platforms = cl.get_platforms()
        platform = platforms[0]
        devices = platform.get_devices()
        device = devices[0]

        ctx = cl.Context([device])
        commQ = cl.CommandQueue(ctx)

        file = open("program_source2.cl", "r")
        program = cl.Program(ctx, file.read())
        program.build()
        img = cv.imread(file_path)
        arrayIn = np.array(img)

        memBuffer = cl.Buffer(ctx, cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_ONLY, hostbuf=arrayIn)
        outputBuffer = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=arrayIn.nbytes)
        width = np.int32(img.shape[0])
        height = np.int32(img.shape[1])
        padding = np.int32(img.strides[0] - img.shape[0] * img.strides[1] * img.itemsize)

        threshold1 = np.float32(1)
        threshold2 = np.float32(100)

        kernelName = program.sobel_threshold
        kernelName.set_arg(0, memBuffer)
        kernelName.set_arg(1, height)
        kernelName.set_arg(2, width)
        kernelName.set_arg(3, padding)
        kernelName.set_arg(4, outputBuffer)
        kernelName.set_arg(5, threshold1)
        kernelName.set_arg(6, threshold2)
        size = width * height
        # Launch the kernel (use a work group size of 10)

        globalWorkSize = ((width + padding), (height + padding))
        workGroupSize = (16, 16)

        kernelEvent = cl.enqueue_nd_range_kernel(queue=commQ, kernel=kernelName, global_work_size=globalWorkSize,
                                                 local_work_size=workGroupSize)

        # Wait for the kernel to finish
        kernelEvent.wait()

        cl.enqueue_copy(commQ, arrayIn, memBuffer)

        new_img = np.empty_like(arrayIn)
        cl.enqueue_copy(commQ, new_img, outputBuffer)

        iF.showSideBySideImages(img, new_img, "", False, False)
        # Release OpenCL resources
        memBuffer.release()
        outputBuffer.release()

    except Exception as e:
        print(e)
        return False
    return True


# compare_times()
# brightness_contrast()
sobel_threshold()
