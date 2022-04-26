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
	MyMesh mymesh1;
	MyMesh mymesh2;
	MyMesh mymesh3;
	MyMesh mymesh4;
	//MyMesh mymesh3;
	string filename = "../example/1/first.stl";
	OpenMesh::IO::read_mesh(mymesh2, filename);
	OpenMesh::IO::read_mesh(mymesh3, filename);

	for (int i = -1; i < 10; i++)
	{
		if (i >= 0){
			angleBasedSmoothing(mymesh2, 1);
			GetMe(mymesh3, 1);
		}
		vector<double> qualities2 = computeMeshAngleQuality(mymesh2);
		vector<double> qualities3 = computeMeshAngleQuality(mymesh3);
		vector<double> qualities5 = computeMeshIdealElementQuality(mymesh2);
		vector<double> qualities6 = computeMeshIdealElementQuality(mymesh3);
		//vector<double> qualities4 = computeMeshAngleQuality(mymesh4);
		//输出网格质量
		cout << "angleBasedSmoothing    第" << i+1 << "次：";
		outMeshAngleQuality(qualities2);
		outMeshIdealElementQuality(qualities5);
		cout << "GetMe      第" << i+1 << "次：";
		outMeshAngleQuality( qualities3);
		outMeshIdealElementQuality(qualities6);
		//cout << "SmartSplitAngle 第" << i+1 << "次：";
		//outMeshAngleQuality(qualities4);
		cout << endl;
		//bool out1 = OpenMesh::IO::write_mesh(mymesh1, "1.stl");
	}

	bool out2 = OpenMesh::IO::write_mesh(mymesh2, "ange2000.stl");
	bool out3 = OpenMesh::IO::write_mesh(mymesh3, "get2000.stl");
}
