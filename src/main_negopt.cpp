#include "smoothing.h"
#include "dataType.h"
void main()
{
	//���Ż������ɳ�ʼ�����������
	MyMesh mesh;
	bool result0 = OpenMesh::IO::read_mesh(mesh, "../example/7/naca0012.stl");
	negprove(mesh,1);
	bool out1 = OpenMesh::IO::write_mesh(mesh, "../example/7/naca0012_new.stl");
}