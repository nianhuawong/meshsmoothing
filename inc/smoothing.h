#ifndef _SMOOTHINGFILE_
#define _SMOOTHINGFILE_
#pragma once
#include <vector>
#include "dataType.h"
#include <Dense>

enum SMOOTH_METHOD
{
	LAPLACIAN = 1,
	ANGLE_BASED = 2,
	GETME = 3,
	NN_BASED = 4,
	DRL_BASED = 5
};

typedef void (*smooth_func_ptr)(MyMesh& mesh, int iter);

smooth_func_ptr get_smooth_func(int method);

void DRL_BasedSmoothing(MyMesh& mesh, int iternum);

void angleBasedSmoothing(MyMesh &mesh, int iternum);
void smartangleBasedSmoothing(MyMesh &mesh, int iternum);

void LaplacianBasedSmoothing(MyMesh& mesh, int iternum);

void GetMe(MyMesh &mesh, int iternum);

void negprove(MyMesh &mesh, int iternum);

Eigen::Vector2d nnopt3(std::vector<MyMesh::Point> ppoint);
Eigen::Vector2d nnopt4(std::vector<MyMesh::Point> ppoint);
Eigen::Vector2d nnopt5(std::vector<MyMesh::Point> ppoint);
Eigen::Vector2d nnopt6(std::vector<MyMesh::Point> ppoint);
Eigen::Vector2d nnopt7(std::vector<MyMesh::Point> ppoint);
Eigen::Vector2d nnopt8(std::vector<MyMesh::Point> ppoint);
Eigen::Vector2d nnopt9(std::vector<MyMesh::Point> ppoint);

void readopt3(char* filename);
void readopt4(char* filename);
void readopt5(char* filename);
void readopt6(char* filename);
void readopt7(char* filename);
void readopt8(char* filename);
void readopt9(char* filename);


void NNSmoothing(MyMesh &mesh, int iternum);





#endif // !_SMOOTHINGFILE_

