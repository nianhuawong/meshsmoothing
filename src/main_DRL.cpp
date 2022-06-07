#include <iostream>
#include <vector>
#include "smoothing.h"
#include "meshquality.h"
#include "writefile.h"
#include "equation.h"
#include "NegaOptimize.h"
#include <graphics.h>      // 引用图形库头文件
#include <conio.h>
#include <cstddef>
#include <cstdlib>
#include <OpenMesh/Core/IO/writer/VTKWriter.hh>
using namespace std;

//enum SMOOTH_METHOD
//{
//	NEGPROVE = 0,
//	LAPLACIAN = 1,
//	ANGLE_BASED = 2,
//	GETME = 3,
//	NN_BASED = 4,
//	DRL_BASED = 5,
//	OPTIMIZATION = 6
//};

void main()
{
	int smooth_method = 5;
	smooth_func_ptr smooth_exec = get_smooth_func(smooth_method);
	int smooth_iterations = 1;

	string filename = "../example/1/first.stl";		//1
	//string filename = "../example/2/second.stl";		
	//string filename = "../example/3/third.stl";
	//string filename = "../example/4/fourth.stl";		//2
	//string filename = "../example/5/fifth.stl";		//4
	//string filename = "../example/6/sixth.stl";		//5
	// 
	//string filename = "C:/Users/86158/Desktop/Todo/quad_medium_pert.stl"; 

	char out_file[256];
	string mainName, extensionName;
	GetFileNameExtension(filename, mainName, extensionName, ".");

	strcpy(out_file, mainName.c_str());
	strcat(out_file, "_smoothed.vtk");

	char outQualityFile[256];
	strcpy(outQualityFile, mainName.c_str());
	strcat(outQualityFile, "_quality_10.txt");

	//ofstream fout;
	//fout.open(outQualityFile, ios::app);
	//fout << smooth_method << " ";
	//fout.close();

	MyMesh mymesh;
	if (extensionName == "stl")
	{
		OpenMesh::IO::read_mesh(mymesh, filename);
	}
	else if (extensionName == "vtk")
	{
		readTriVTK(mymesh, filename.c_str());
	}
	
	int smoothOrNot = smooth_method >= NEGPROVE && smooth_iterations > 0;
	for (int i = 0; i < smoothOrNot; i++)
	{
		clock_t ts = clock();
		if (i >= 0)
		{
			smooth_exec(mymesh, smooth_iterations);
		}
		clock_t te = clock();

		vector<double> qualities1 = computeMeshAngleQuality(mymesh);
		vector<double> qualities2 = computeMeshIdealElementQuality(mymesh);
		double minAngle = computeMeshAveang(mymesh);

		cout << "==========================" << endl;
		int iter = (i > 0) ? smooth_iterations : (i + 1);
		cout << "第" << iter << "次：\n";
		outMeshAngleQuality(qualities1);
		outMeshIdealElementQuality(qualities2);
		cout << "Time elapsed " << (te - ts) / 1000. << " sec." << endl;
		cout << endl;

		outFileMeshAngleQuality(outQualityFile, qualities1);
		outFileMeshIdealElementQuality(outQualityFile, qualities2);
	}

	writeVTKWithFixedBoundery(mymesh, out_file);

	//OpenMesh::IO::_VTKWriter_  vtkWriter = OpenMesh::IO::VTKWriter();
	//bool out1 = OpenMesh::IO::write_mesh(mymesh, out_file);
	//bool out2 = OpenMesh::IO::write_mesh(mymesh, out_file);
}
