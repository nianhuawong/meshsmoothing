#include <iostream>
#include <vector>
#include "smoothing.h"
#include "meshquality.h"
#include "writefile.h"
#include "equation.h"
#include "NegaOptimize.h"
#include <graphics.h>      // 引用图形库头文件
#include <conio.h>
#include "dataType.h"
using namespace std;

void main()
{
	readopt3("opt3.txt");
	readopt4("opt4.txt");
	readopt5("opt5.txt");
	readopt6("opt6.txt");
	readopt7("opt7.txt");
	readopt8("opt8.txt");
	readopt9("opt9.txt");
	MyMesh mesh1, mesh2, mesh3;
	bool result0 = OpenMesh::IO::read_mesh(mesh1, "e2000.stl");
	bool result1 = OpenMesh::IO::read_mesh(mesh2, "e2000.stl");
	bool result2 = OpenMesh::IO::read_mesh(mesh3, "e2000.stl");

	clock_t ts = clock();
	GetMe(mesh1, 10);
	clock_t te = clock();
	cout << 0 << " tests " << (te - ts) / 1000. << " sec." << endl;
	OpenMesh::IO::write_mesh(mesh1, "1111.stl");


	ts = clock();
	//laplacianSmoothing(mesh2, 10);
	te = clock();
	cout << 1 << " tests " << (te - ts) / 1000. << " sec." << endl;
	OpenMesh::IO::write_mesh(mesh2, "11111.stl");


	ts = clock();
	NNSmoothing(mesh3, 10);
	te = clock();
	cout << 2 << " tests " << (te - ts) / 1000. << " sec." << endl;
	OpenMesh::IO::write_mesh(mesh3, "nne2000.stl");
	//bool out1 = OpenMesh::IO::write_mesh(mesh3, "shenjingwangluo.stl");
	cout << "hello" << endl;
}