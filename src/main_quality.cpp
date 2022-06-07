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
int main()
{
	//string filename = "../example/1/first_opt_smoothed.vtk";
	string filename = "../example/2/second.stl";		
	//string filename = "../example/3/third.stl";

	char out_file[256];
	string mainName, extensionName;
	GetFileNameExtension(filename, mainName, extensionName, ".");

	MyMesh mymesh;
	if (extensionName == "stl")
	{
		if (!OpenMesh::IO::read_mesh(mymesh, filename)) return 1;
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
	cout << "网格质量检查......" << endl;
	outMeshAngleQuality(qualities1);
	outMeshIdealElementQuality(qualities2);	
	cout << endl;

	//================================================================
	GetFileNameExtension(filename, mainName, extensionName, "/");

	char outQualityFile[256];
	strcpy(outQualityFile, mainName.c_str());
	strcat(outQualityFile, "/quality.txt");

	//outFileMeshAngleQuality(outQualityFile, qualities1);
	//outFileMeshIdealElementQuality(outQualityFile, qualities2);

	return 0;
}
