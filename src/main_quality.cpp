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
void main()
{
	string filename = "../example/1/first_opt_smoothed.vtk";		//1

	char out_file[256];
	string mainName, extensionName;
	GetFileNameExtension(filename, mainName, extensionName, ".");

	MyMesh mymesh;
	if (extensionName == "stl")
	{
		OpenMesh::IO::read_mesh(mymesh, filename);
	}
	else if (extensionName == "vtk")
	{
		readTriVTK(mymesh, filename.c_str());
	}

	vector<double> qualities1 = computeMeshAngleQuality(mymesh);
	vector<double> qualities2 = computeMeshIdealElementQuality(mymesh);
	double minAngle = computeMeshAveang(mymesh);
	//vector<double> qualities23 = computeMeshAreaRatioQuality(mymesh);

	cout << "==========================" << endl;
	cout << "网格质量检查......";
	outMeshAngleQuality(qualities1);
	outMeshIdealElementQuality(qualities2);	
	cout << endl;
}
