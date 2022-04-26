#include <iostream>
#include <vector>
#include "smoothing.h"
#include "meshquality.h"
#include "writefile.h"
#include "equation.h"
#include "NegaOptimize.h"
#include <graphics.h>      // 引用图形库头文件
#include <conio.h>
using namespace std;

void main(){
	// 测试比较优化算法
	MyMesh mesh0;
	MyMesh mesh1;
	MyMesh mesh2;
	MyMesh mesh3;
	bool result0 = OpenMesh::IO::read_mesh(mesh0, "getsresult3.stl");
	bool result1 = OpenMesh::IO::read_mesh(mesh1, "myoptsresult3.stl");
	//bool result2 = OpenMesh::IO::read_mesh(mesh2, "foot.stl");
	//bool result3 = OpenMesh::IO::read_mesh(mesh3, "foot_eas.stl");
	if (result1 == false || result1 == false)
	{
		return;
	}
	vector<double> qualities0 = computeMeshAngleQuality(mesh0);
	vector<double> qualities1 = computeMeshAngleQuality(mesh1);
	vector<double> qualities2 = computeMeshIdealElementQuality(mesh0);
	vector<double> qualities3 = computeMeshIdealElementQuality(mesh1);
	//vector<double> qualities3 = computeMeshIdealElementQuality(mesh3);
	outMeshAngleQuality(qualities0);
	outMeshIdealElementQuality(qualities2);
	outMeshAngleQuality(qualities1);
	outMeshIdealElementQuality(qualities3);
}
