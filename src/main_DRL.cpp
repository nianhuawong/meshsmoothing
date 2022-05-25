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
#include "evaluatePolicy.h"
#include "evaluatePolicy_terminate.h"
using namespace std;

// Function Declarations
static void argInit_10x2_real_T(double result[20]);
static double argInit_real_T();

// Function Definitions

//
// Arguments    : double result[20]
// Return Type  : void
//
static void argInit_10x2_real_T(double result[20])
{
	double result_tmp;

	// Loop over the array to initialize each element.
	result_tmp = argInit_real_T();
	for (int idx0 = 0; idx0 < 10; idx0++) {
		// Set the value of the array element.
		// Change this value to the value that the application requires.
		result[idx0] = result_tmp;

		// Set the value of the array element.
		// Change this value to the value that the application requires.
		result[idx0 + 10] = result_tmp;
	}
}

//
// Arguments    : void
// Return Type  : double
//
static double argInit_real_T()
{
	return 1.0;
}

void DRL_BasedSmoothing(MyMesh& mesh, int iternum);

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

void DRL_BasedSmoothing(MyMesh& mesh, int iternum)
{
	for (int i = 0; i < iternum; i++)
	{
		for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); v_it++)
		{
			MyMesh::Point cog;
			MyMesh::Scalar valence;
			MyMesh::Point position;
			vector<MyMesh::Point>  oldpoints;

			if (mesh.is_boundary(*v_it))
				continue;

			position[2] = 0.0;
			cog[0] = cog[1] = cog[2] = valence = 0.0;

			//找出ring的所有点来
			MyMesh::Scalar ringxmin = mesh.point(*v_it)[0];
			MyMesh::Scalar ringxmax = mesh.point(*v_it)[0];
			MyMesh::Scalar ringymin = mesh.point(*v_it)[1];
			MyMesh::Scalar ringymax = mesh.point(*v_it)[1];
			for (auto vv_it = mesh.vv_ccwbegin(*v_it); vv_it != mesh.vv_ccwend(*v_it); ++vv_it)
			{
				oldpoints.push_back(mesh.point(*vv_it));
				cog += mesh.point(*vv_it);
				++valence;

				if (ringxmin > mesh.point(*vv_it)[0]) ringxmin = mesh.point(*vv_it)[0];
				if (ringxmax < mesh.point(*vv_it)[0]) ringxmax = mesh.point(*vv_it)[0];
				if (ringymin > mesh.point(*vv_it)[1]) ringymin = mesh.point(*vv_it)[1];
				if (ringymax < mesh.point(*vv_it)[1]) ringymax = mesh.point(*vv_it)[1];
			}

			//归一化到0-1之间
			MyMesh::Scalar len;
			if (ringxmax - ringxmin > ringymax - ringymin)
				len = ringxmax - ringxmin;
			else
				len = ringymax - ringymin;

			for (int i = 0; i < oldpoints.size(); i++)
			{
				oldpoints[i][0] = (oldpoints[i][0] - ringxmin) / len;
				oldpoints[i][1] = (oldpoints[i][1] - ringymin) / len;
			}

			//将ring nodes加入observation数组
			vector<double> observation(20);
			for (int i = 0; i < oldpoints.size(); i++)
			{
				observation[i     ] = oldpoints[i][0];
				observation[i + 10] = oldpoints[i][1];
			}

			//补齐10个ring nodes
			for (int i = oldpoints.size(); i < 10; i++)
			{
				observation[i     ] = 0.0;
				observation[i + 10] = 0.0;
			}

			//优化
			vector<float> action(2);
			evaluatePolicy(&observation[0], &action[0]);

			Eigen::Vector2d vout(action[0], action[1]);

			//映射回去
			position[0] = vout[0];
			position[1] = vout[1];
			position[0] = position[0] * len + ringxmin;
			position[1] = position[1] * len + ringymin;

			mesh.set_point(*v_it, position);
		}
	}

}
