#include "smoothing.h"
#include "dataType.h"
void main()
{
	//负优化，生成初始质量差的网格
	MyMesh mesh;
	bool result0 = OpenMesh::IO::read_mesh(mesh, "../example/7/naca0012.stl");
	negprove(mesh,1);
	bool out1 = OpenMesh::IO::write_mesh(mesh, "../example/7/naca0012_new.stl");
}