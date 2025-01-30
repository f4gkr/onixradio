#include <iostream>
#include <string>
#include <assert.h>
#include <png.h>
#include <array>
#include <vector>
#include <cmath>
#include <algorithm>

#include "onnxruntime_c_api.h"
#include "providers.h"

const OrtApi *g_ort = nullptr;

#define ORT_ABORT_ON_ERROR(expr)                                   \
    do                                                             \
    {                                                              \
        OrtStatus *onnx_status = (expr);                           \
        if (onnx_status != nullptr)                                \
        {                                                          \
            const char *msg = g_ort->GetErrorMessage(onnx_status); \
            std::cout << msg << std::endl;                         \
            g_ort->ReleaseStatus(onnx_status);                     \
            abort();                                               \
        }                                                          \
    } while (0);

int main(int argc, char *argv[])
{
    size_t count;
    OrtEnv *env;
    OrtSessionOptions *session_options;
    OrtSession *session;
    OrtMemoryInfo *memory_info;
    size_t model_input_ele_count = 2048;

    ORTCHAR_T *model_path = argv[1];
    if( model_path == nullptr ) {
        model_path = (const char *)"complexmult.onnx" ;
    }

    fprintf(stdout, "Using ONNX file %s\n", (char *)argv[1]);
    fflush(stdout);

    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    ORT_ABORT_ON_ERROR(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env));
    ORT_ABORT_ON_ERROR(g_ort->CreateSessionOptions(&session_options));
    ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, model_path, session_options, &session));
    ORT_ABORT_ON_ERROR(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));

    ORT_ABORT_ON_ERROR(g_ort->SessionGetInputCount(session, &count));
    fprintf(stdout, "Model has %d input tensors\n", (int)count);
    ORT_ABORT_ON_ERROR(g_ort->SessionGetOutputCount(session, &count));
    fprintf(stdout, "Model has %d output tensors\n", (int)count);

    float *A = (float *)malloc(model_input_ele_count * sizeof(float));
    float *B = (float *)malloc(model_input_ele_count * sizeof(float));
    memset( A, 0, model_input_ele_count * sizeof(float));
    memset( B, 0, model_input_ele_count * sizeof(float));
    for( int i=0 ; i < 1024 ; i++ ) {
         B[2*i] = 1.0f ;
         A[2*i] = (float)i ;
    }

    float *R = (float *)malloc(model_input_ele_count * sizeof(float));
    memset( R, 0, model_input_ele_count * sizeof(float));


    const int64_t input_shape[] = {(int64_t)model_input_ele_count};
    const size_t input_shape_len = sizeof(input_shape) / sizeof(input_shape[0]);
    const size_t model_input_len = model_input_ele_count * sizeof(float);

    OrtValue *input_tensor_A = NULL;
    OrtValue *input_tensor_B = NULL;
    ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, A, model_input_len, input_shape,
                                                             input_shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                             &input_tensor_A));
    ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, B, model_input_len, input_shape,
                                                             input_shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                             &input_tensor_B));
    int is_tensor;
    ORT_ABORT_ON_ERROR(g_ort->IsTensor(input_tensor_A, &is_tensor));
    assert(is_tensor);
    ORT_ABORT_ON_ERROR(g_ort->IsTensor(input_tensor_B, &is_tensor));
    assert(is_tensor);

    // g_ort->ReleaseMemoryInfo(memory_info);
    const char *input_names[] = {"A", "B"};
    const char *output_names[] = {"out"};

    const int64_t output_shape[] = {(int64_t)model_input_ele_count};
    const size_t output_shape_len = sizeof(output_shape) / sizeof(output_shape[0]);
    const size_t model_output_len = model_input_ele_count * sizeof(float);

    OrtValue *output_tensor = NULL;
    ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, R, model_output_len, output_shape,
                                                             input_shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                             &output_tensor));
     OrtValue *inputs[2] ;
     inputs[0] = input_tensor_A ;
     inputs[1] = input_tensor_B ;

    ORT_ABORT_ON_ERROR(
        g_ort->Run(session, NULL, input_names, (const OrtValue *const *)&inputs, 2, output_names, 1, &output_tensor));

    assert(output_tensor != NULL);
    ORT_ABORT_ON_ERROR(g_ort->IsTensor(output_tensor, &is_tensor));
    assert(is_tensor);

    for( int i=0 ; i < 1024 ; i++ ) {
         float I = R[2*i] ;
         float Q = R[2*i+1];
         fprintf( stdout, "%.3f %.3f\n", I, Q); fflush(stdout);
    }
}