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
using namespace std;

void main()
{
	MyMesh mymesh1;
	string filename = "../example/1/first.stl";
	OpenMesh::IO::read_mesh(mymesh1, filename);

	for (int i = -1; i < 1; i++)
	{
		if (i >= 0)
		{
			DRL_BasedSmoothing(mymesh1, 1);			
		}

		vector<double> qualities1 = computeMeshAngleQuality(mymesh1);
		vector<double> qualities2 = computeMeshIdealElementQuality(mymesh1);

		cout << "DRL_BasedSmoothing    第" << i+1 << "次：";
		outMeshAngleQuality(qualities1);
		outMeshIdealElementQuality(qualities2);
		cout << endl;
	}

	bool out2 = OpenMesh::IO::write_mesh(mymesh1, "C:/Users/86158/Desktop/Todo/20220525.stl");
}
