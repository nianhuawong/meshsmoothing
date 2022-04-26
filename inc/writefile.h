#ifndef _WRITEFILE_
#define _WRITEFILE_
#include <iostream>
#include <vector>
using namespace std;
// -------------------- OpenMesh
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
typedef OpenMesh::TriMesh_ArrayKernelT<>  MyMesh;

// -------------------- OpenVolumeMesh
#include<OpenVolumeMesh/Mesh/TetrahedralMesh.hh>
typedef OpenVolumeMesh::GeometricTetrahedralMeshV3d MyTetMesh;
typedef OpenVolumeMesh::Geometry::Vec3d         Vec3d;
// ----------------------------------------------------------------------------

//�����ı���
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>
typedef OpenMesh::PolyMesh_ArrayKernelT<>  PolyMesh;

void writeVTK(MyMesh &mesh, char* filename);
void writeVTKWithFixedBoundery(MyMesh &mesh, char* filename);

void readEleFile(MyTetMesh &mytet, string filename);
//����������vtk��ʽ
bool readTetVTK(MyTetMesh &mesh, string fname);

//�����ı�������
void readQuadVTK(PolyMesh &mesh, char* filename);

//����ı�������
void writeQuadVTK(PolyMesh &mesh, char* filename);


//��������������
void readTriVTK(MyMesh &mesh, char* filename);

//�������������
void writeTriVTK(MyMesh &mesh, char* filename);

//��������������
void readTritxt(MyMesh &mesh, char* filename1, char* filename2);

void readliu(MyMesh &mesh, char* filename);

#endif