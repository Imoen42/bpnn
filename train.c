#include "bpnn.h"
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>

float error;
float f = 0.0001;//误差精度
typedef struct
{
    float** x;//训练样本,二维动态数组
    float** y;//样本理想输出
    int in;
    int out;
    int rows;//文件行数
}
Data;

//统计文件行数
static int lns(FILE* const file)
{
    int ch = EOF;
    int lines = 0;
    int pc = '\n';
    while((ch = getc(file)) != EOF)
    {
        if(ch == '\n')
            lines++;
        pc = ch;
    }
    if(pc != '\n')
        lines++;
    rewind(file);
    return lines;
}

static char* readln(FILE* const file)
{
    int ch = EOF;
    int reads = 0;
    int size = 128;
    char* line = (char*) malloc((size) * sizeof(char));
    while((ch = getc(file)) != '\n' && ch != EOF)
    {
        line[reads++] = ch;
        if(reads + 1 == size)
            line = (char*) realloc((line), (size *= 2) * sizeof(char));
    }
    line[reads] = '\0';
    return line;
}

static float** new2d(const int rows, const int cols)
{
    float** row = (float**) malloc((rows) * sizeof(float*));
    for(int r = 0; r < rows; r++)
        row[r] = (float*) malloc((cols) * sizeof(float));
    return row;
}

static Data ndata(const int in, const int out, const int rows)
{
    const Data data = {
        new2d(rows, in), new2d(rows, out), in, out, rows
    };
    return data;
}

static void parse(const Data data, char* line, const int row)
{
    const int cols = data.in + data.out;
    for(int col = 0; col < cols; col++)
    {
        const float val = atof(strtok(col == 0 ? line : NULL, " "));
        if(col < data.in)
            data.x[row][col] = val;
        else
            data.y[row][col - data.in] = val;
    }
}

static void dfree(const Data d)
{
    for(int row = 0; row < d.rows; row++)
    {
        free(d.x[row]);
        free(d.y[row]);
    }
    free(d.x);
    free(d.y);
}

static void shuffle(const Data d)
{
    for(int a = 0; a < d.rows; a++)
    {
        const int b = rand() % d.rows;
        float* ot = d.y[a];
        float* it = d.x[a];
        d.y[a] = d.y[b];
        d.y[b] = ot;
        d.x[a] = d.x[b];
        d.x[b] = it;
    }
}

static Data build(const char* path, const int in, const int out)
{
    FILE* file = fopen(path, "r");
    if(file == NULL)
    {
        printf("404 %s\n", path);
        exit(1);
    }
    const int rows = lns(file);
    Data data = ndata(in, out, rows);
    for(int row = 0; row < rows; row++)
    {
        char* line = readln(file);
        parse(data, line, row);
        free(line);
    }
    fclose(file);
    return data;
}

int main()
{
    srand(time(0));
    const int in = 4;
    const int out = 2;
    float rate = 1.0f;//学习率
    const int hidden_layers = 28;
    const float anneal = 0.99f;
    const int iterations = 1000;//最大循环次数
	error = f + 1;
	int i;
    const Data data = build("train.data", in, out);
    const bpnn bp = bp_build(in, hidden_layers, out);
    for(i = 0; error / data.rows > f && i < iterations; i++)
    {
        shuffle(data);
        error = 0.0f;
        for(int j = 0; j < data.rows; j++)
        {
            const float* const x = data.x[j];
            const float* const y = data.y[j];
            error += bp_train(bp, x, y, rate);
        }
		if (i % 10 == 0)
        	printf("误差: %.12f :: 学习率: %f\n",
            	(double) error / data.rows,
            	(double) rate);
        rate *= anneal;
    }
	printf("设置循环次数：%d 实际循环次数：%d\n", iterations,i);
    bp_save(bp, "saved.data");
    bp_free(bp);
    dfree(data);
    return 0;
}
