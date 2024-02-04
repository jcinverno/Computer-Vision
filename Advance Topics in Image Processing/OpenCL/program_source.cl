__kernel void multiply(__global int* array, const int constant_number, __global int* sum_result) {

    int global_id = get_global_id(0);

    array[global_id] = array[global_id] * constant_number;

    atomic_add(sum_result, array[global_id]);
}
