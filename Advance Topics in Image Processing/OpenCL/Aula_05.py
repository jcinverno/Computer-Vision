import pyopencl as cl
import imageForms as iF
import numpy as np


def get_platform_and_device_information():
    platforms = cl.get_platforms()

    for platform in platforms:
        name = platform.get_info(cl.platform_info.NAME)
        vendor = platform.get_info(cl.platform_info.VENDOR)
        version = platform.get_info(cl.platform_info.VERSION)

        displayStr = "Name: " + name + "\nVendor: " + vendor + "\nVersion: " + version + "\n"
        iF.showMessageBox(title="Platform Info", message=displayStr)

        devices = platform.get_devices()

        for device in devices:
            displayStr = "VENDOR: " + device.get_info(cl.device_info.VENDOR)
            displayStr = displayStr + "\nNAME: " + device.get_info(cl.device_info.NAME)
            displayStr = displayStr + "\nMAX_COMPUTE_UNITS: " + str(device.get_info(cl.device_info.MAX_COMPUTE_UNITS))
            displayStr = displayStr + "\nMAX_WORK_ITEM_DIMENSIONS: " + str(
                device.get_info(cl.device_info.MAX_WORK_ITEM_DIMENSIONS))
            displayStr = displayStr + "\nMAX_WORK_ITEM_SIZES: " + str(
                device.get_info(cl.device_info.MAX_WORK_ITEM_SIZES))
            displayStr = displayStr + "\nMAX_WORK_GROUP_SIZE: " + str(
                device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE))
            displayStr = displayStr + "\nMAX_CONSTANT_ARGS: " + str(device.get_info(cl.device_info.MAX_CONSTANT_ARGS))
            displayStr = displayStr + "\nIMAGE_SUPPORT: " + str(device.get_info(cl.device_info.IMAGE_SUPPORT))
            displayStr = displayStr + "\nIMAGE2D_MAX_WIDTH: " + str(device.get_info(cl.device_info.IMAGE2D_MAX_WIDTH))
            displayStr = displayStr + "\nIMAGE2D_MAX_HEIGHT: " + str(device.get_info(cl.device_info.IMAGE2D_MAX_HEIGHT))
            displayStr = displayStr + "\nLOCAL_MEM_SIZE: " + str(device.get_info(cl.device_info.LOCAL_MEM_SIZE))
            displayStr = displayStr + "\nPREFERRED_WORK_GROUP_SIZE_MULTIPLE: " + str(
                device.get_info(cl.device_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE))
            iF.showMessageBox(title="Device Info", message=displayStr)


def multiply_vector():
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

        arrayIn = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=int)

        memBuffer = cl.Buffer(ctx, cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_WRITE, hostbuf=arrayIn)
        sumBuffer = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=4)

        # Set kernel arguments
        kernelName = program.multiply
        kernelName.set_arg(0, memBuffer)
        kernelName.set_arg(1, np.int32(2))
        kernelName.set_arg(2, sumBuffer)

        # Launch the kernel (use a work group size of 10)
        globalWorkSize = (10,)
        workGroupSize = (10,)

        kernelEvent = cl.enqueue_nd_range_kernel(queue=commQ, kernel=kernelName, global_work_size=globalWorkSize,
                                                 local_work_size=workGroupSize)

        # Wait for the kernel to finish
        kernelEvent.wait()

        # Copy the array and sum from the device to the host
        cl.enqueue_copy(commQ, arrayIn, memBuffer)
        sum_result = np.zeros(1, dtype=np.int32)
        cl.enqueue_copy(commQ, sum_result, sumBuffer)

        print("Modified Array:", arrayIn)
        print("Sum of Elements:", sum_result[0])

        # Release OpenCL resources
        memBuffer.release()
        sumBuffer.release()

    except Exception as e:
        print(e)
        return False
    return True


multiply_vector()
# get_platform_and_device_information()
