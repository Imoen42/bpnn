#include "bpnn.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//损失函数
static float err(const float a, const float b)
{
    return 0.5f * (a - b) * (a - b);
}

static float bp_err(const float* const y, const float* const o, const int size)
{
    float sum = 0.0f;
    for(int i = 0; i < size; i++)
        sum += err(y[i], o[i]);
    return sum;
}

//激活函数
static float sigmoid(const float a)
{
	return 1.0f / (1.0f + expf(-a));
}

//返回0.0-1.0之间的随机浮点数
static float frand()
{
    return rand() / (float) RAND_MAX;
}

//反向传播
static void bprop(const bpnn t, const float* const x, const float* const y, float rate)
{
    for(int i = 0; i < t.hidden_layers; i++)
    {
        float sum = 0.0f;
		//计算相对于输出的总误差变化
        for(int j = 0; j < t.out; j++)
        {
			const float a = t.o[j] - y[j];
			const float b = t.o[j] * (1.0f - t.o[j]);
            sum += a * b * t.output_weights[j * t.hidden_layers + i];
			//修改输出层权值
            t.output_weights[j * t.hidden_layers + i] -= rate * a * b * t.h[i];
        }
		//修改隐藏层权值
        for(int j = 0; j < t.in; j++)
			t.all_weights[i * t.in + j] -= rate * sum * t.h[i] * (1.0f - t.h[i]) * x[j];
    }
}

//前向传播
static void fprop(const bpnn t, const float* const x)
{
	//计算隐藏层神经元值
    for(int i = 0; i < t.hidden_layers; i++)
    {
        float sum = 0.0f;
        for(int j = 0; j < t.in; j++)
            sum += x[j] * t.all_weights[i * t.in + j];
        t.h[i] = sigmoid(sum + t.b[0]);
    }
	//计算输出层神经元值
    for(int i = 0; i < t.out; i++)
    {
        float sum = 0.0f;
        for(int j = 0; j < t.hidden_layers; j++)
            sum += t.h[j] * t.output_weights[i * t.hidden_layers + j];
        t.o[i] = sigmoid(sum + t.b[1]);
    }
}

static void wbrand(const bpnn t)
{
    for(int i = 0; i < t.nw; i++) t.all_weights[i] = frand() - 0.5f;
    for(int i = 0; i < t.nb; i++) t.b[i] = frand() - 0.5f;
}

//返回给定输入的输出预测
float* bp_predict(const bpnn t, const float* const x)
{
    fprop(t, x);
    return t.o;
}

float bp_train(const bpnn t, const float* const x, const float* const y, float rate)
{
    fprop(t, x);
    bprop(t, x, y, rate);
    return bp_err(y, t.o, t.out);
}

bpnn bp_build(const int in, const int hidden_layers, const int out)
{
    bpnn t;
    t.nb = 2;
    t.nw = hidden_layers * (in + out);
    t.all_weights = (float*) calloc(t.nw, sizeof(*t.all_weights));
    t.output_weights = t.all_weights + hidden_layers * in;
    t.b = (float*) calloc(t.nb, sizeof(*t.b));
    t.h = (float*) calloc(hidden_layers, sizeof(*t.h));
    t.o = (float*) calloc(out, sizeof(*t.o));
    t.in = in;
    t.hidden_layers = hidden_layers;
    t.out = out;
    wbrand(t);
    return t;
}

//数据保存到文件
void bp_save(const bpnn t, const char* const path)
{
    FILE* const file = fopen(path, "w");
    fprintf(file, "%d %d %d\n", t.in, t.hidden_layers, t.out);
    for(int i = 0; i < t.nb; i++) fprintf(file, "%f\n", (double) t.b[i]);
    for(int i = 0; i < t.nw; i++) fprintf(file, "%f\n", (double) t.all_weights[i]);
    fclose(file);
}

//从文件读取数据
bpnn bp_load(const char* const path)
{
    FILE* const file = fopen(path, "r");
    int in = 0;
    int hidden_layers = 0;
    int out = 0;
    fscanf(file, "%d %d %d\n", &in, &hidden_layers, &out);
    const bpnn t = bp_build(in, hidden_layers, out);
    for(int i = 0; i < t.nb; i++) fscanf(file, "%f\n", &t.b[i]);
    for(int i = 0; i < t.nw; i++) fscanf(file, "%f\n", &t.all_weights[i]);
    fclose(file);
    return t;
}

//释放内存
void bp_free(const bpnn t)
{
    free(t.all_weights);
    free(t.b);
    free(t.h);
    free(t.o);
}

void bp_print(const float* arr, const int size)
{
    for(int i = 0; i < size; i++)
        printf("%f ", (double) arr[i]);
    printf("\n");
}
