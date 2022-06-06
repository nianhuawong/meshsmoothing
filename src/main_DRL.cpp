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
//	LAPLACIAN = 1,
//	ANGLE_BASED = 2,
//	GETME = 3,
//	NN_BASED = 4,
//	DRL_BASED = 5
//};

void main()
{
	int smooth_method = DRL_BASED;
	smooth_func_ptr smooth_exec = get_smooth_func(smooth_method);
	int smooth_iterations = 2;

	//string filename = "C:/Users/86158/Desktop/Todo/quad_medium_pert.stl";
	string filename = "../example/1/first.stl";		//1
	//string filename = "../example/4/fourth.stl";		//2
	//string filename = "../example/5/fifth.stl";		//4
	//string filename = "../example/6/sixth.stl";		//5
	//string filename = "../example/2/second.stl";		//

	char out_file[256];
	string mainName, extensionName;
	GetFileNameExtension(filename, mainName, extensionName, ".");

	strcpy(out_file, mainName.c_str());
	strcat(out_file, "_smoothed.vtk");

	MyMesh mymesh;
	if (extensionName == "stl")
	{
		OpenMesh::IO::read_mesh(mymesh, filename);
	}
	else if (extensionName == "vtk")
	{
		readTriVTK(mymesh, filename.c_str());
	}
	
	int smoothOrNot = smooth_iterations > 0;
	for (int i = -1; i < smoothOrNot; i++)
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
		cout << "第" << smooth_iterations << "次：\n";
		outMeshAngleQuality(qualities1);
		outMeshIdealElementQuality(qualities2);
		cout << "Time elapsed " << (te - ts) / 1000. << " sec." << endl;
		cout << endl;
	}

	writeVTKWithFixedBoundery(mymesh, out_file);

	//OpenMesh::IO::_VTKWriter_  vtkWriter = OpenMesh::IO::VTKWriter();
	//bool out1 = OpenMesh::IO::write_mesh(mymesh, out_file);
	//bool out2 = OpenMesh::IO::write_mesh(mymesh, out_file);
}
