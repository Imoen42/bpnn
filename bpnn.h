#pragma once

typedef struct
{
    float* all_weights;//所有的权值
    float* output_weights;//隐藏层到输出层权值
    float* b;//偏差
    float* h;//隐藏层
    float* o;//输出层
    int nb;//偏差数量
    int nw;//权值数量
    int in;//输入向量维数
    int hidden_layers;//隐藏层神经元数量
    int out;//输出向量维数
}
bpnn;

float* bp_predict(bpnn, const float* in);

float bp_train(bpnn, const float* x, const float* y, float rate);

bpnn bp_build(int in, int hidden_layers, int out);

void bp_save(bpnn, const char* path);

bpnn bp_load(const char* path);

void bp_free(bpnn);

void bp_print(const float* arr, const int size);
