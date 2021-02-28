#include "bpnn.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static float** x;
static float** y;
static int samples;
static int out = 2;
static int in = 4;

void load_samples()
{
	int i,j,k;
	FILE *fp = fopen("test.data", "r");
	if (!fp) {
		printf("404\n");
		exit(0);
	}
	while (!feof(fp)) {
		if (fgetc(fp) == '\n')
			++samples;
	}
    fseek(fp, 0, SEEK_SET);
    printf("加载样本:%d\n", samples);
    x = (float**)malloc(sizeof(float*) * samples);//第一维
	y = (float**)malloc(sizeof(float*) * samples);
	for (i = 0; i < samples; i++) {
		x[i] = (float*)malloc(in * sizeof(float));//第二维
		y[i] = (float*)malloc(out * sizeof(float));
		for (j = 0; j < in + out; j++) {
			if (j < in) {
				fscanf(fp, "%f", &x[i][j]);
			}
			else {
				k = j - in;
				fscanf(fp, "%f", &y[i][k]);
			}
		}
	}
	fclose(fp);
}

void free_array()
{
	int i;
	for (i = 0; i < samples; i++) {
		free(x[i]);//释放第二维指针
		free(y[i]);
	}
	free(x);//释放第一维指针
	free(y);
}

int main()
{
	load_samples();
	const bpnn loaded = bp_load("saved.data");
	for(int i = 0; i < samples; i++)
	{
		const float* const x1 = x[i];
		const float* const y1 = y[i];
		const float* const pd = bp_predict(loaded, x1);
		printf("样本%d：", i);
		bp_print(x1, in);
		printf("实际输出：");
		bp_print(y1, out);
		printf("预测输出：");
		bp_print(pd, out);
	}
	bp_free(loaded);
	free_array();
	return 0;
}
