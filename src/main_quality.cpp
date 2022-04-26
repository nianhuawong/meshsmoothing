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
	MyMesh mesh;
	bool result0 = OpenMesh::IO::read_mesh(mesh, "sresult3.stl");
	double aveang = computeMeshAveang(mesh);
	cout << aveang << endl;
	MyMesh mesh2;
	bool result2 = OpenMesh::IO::read_mesh(mesh2, "angsresult3.stl");
	double aveang2 = computeMeshAveang(mesh2);
	cout << aveang2 << endl;
	MyMesh mesh3;
	bool result3 = OpenMesh::IO::read_mesh(mesh3, "getsresult3.stl");
	double aveang3 = computeMeshAveang(mesh3);
	cout << aveang3 << endl;
	MyMesh mesh4;
	bool result4 = OpenMesh::IO::read_mesh(mesh4, "optsresult3.stl");
	double aveang4 = computeMeshAveang(mesh4);
	cout << aveang4 << endl;
	MyMesh mesh5;
	bool result5 = OpenMesh::IO::read_mesh(mesh5, "myoptsresult3.stl");
	double aveang5 = computeMeshAveang(mesh5);
	cout << aveang5 << endl;
}
