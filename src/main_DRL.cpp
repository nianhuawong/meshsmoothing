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

enum SMOOTH_METHOD
{
	LAPLACIAN=1,
	ANGLE_BASED=2,
	GETME=3,
	NN_BASED=4,
	DRL_BASED=5
};

typedef void (*smooth_func_ptr)(MyMesh &mesh, int iter);

smooth_func_ptr get_smooth_func(int method);
 
void main()
{
	int smooth_method = DRL_BASED;
	string filename = "C:/Users/86158/Desktop/Todo/quad_medium_pert2.stl";
	//string filename = "../example/1/first.stl";		//1
	//string filename = "../example/4/fourth.stl";		//2
	//string filename = "../example/5/fifth.stl";		//4
	//string filename = "../example/6/sixth.stl";		//5

	smooth_func_ptr smooth_exec = get_smooth_func(smooth_method);

	MyMesh mymesh;
	OpenMesh::IO::read_mesh(mymesh, filename);

	for (int i = -1; i < 1; i++)
	{
		if (i >= 0)
		{
			smooth_exec(mymesh, 1);
		}

		vector<double> qualities1 = computeMeshAngleQuality(mymesh);
		vector<double> qualities2 = computeMeshIdealElementQuality(mymesh);
		double minAngle = computeMeshAveang(mymesh);
		//vector<double> qualities23 = computeMeshAreaRatioQuality(mymesh);

		cout << "==========================" << endl;
		cout << "第" << i + 1 << "次：\n";
		outMeshAngleQuality(qualities1);
		outMeshIdealElementQuality(qualities2);
		cout << endl;
	}

	OpenMesh::IO::_VTKWriter_  vtkWriter = OpenMesh::IO::VTKWriter();
	bool out1 = OpenMesh::IO::write_mesh(mymesh, "C:/Users/86158/Desktop/Todo/first_opt.vtk");
	bool out2 = OpenMesh::IO::write_mesh(mymesh, "C:/Users/86158/Desktop/Todo/first_opt.stl");
}

smooth_func_ptr get_smooth_func(int method)
{
	switch (method)
	{
	case LAPLACIAN:
		return &LaplacianBasedSmoothing;
		break;
	case ANGLE_BASED:
		return &angleBasedSmoothing;
		break;
	case GETME:
		return &GetMe;
		break;
	case NN_BASED:
		readopt3("../opt/opt3.txt");
		readopt4("../opt/opt4.txt");
		readopt5("../opt/opt5.txt");
		readopt6("../opt/opt6.txt");
		readopt7("../opt/opt7.txt");
		readopt8("../opt/opt8.txt");
		readopt9("../opt/opt9.txt");

		return &NNSmoothing;
		break;
	case DRL_BASED:
		return &DRL_BasedSmoothing;
		break;
	default:
		break;
	}
}