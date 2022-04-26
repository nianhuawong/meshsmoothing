#ifndef _WRITEFILE_
#define _WRITEFILE_
#pragma once
#include "dataType.h"

// ----------------------------------------------------------------------------

void writeVTK(MyMesh& mesh, char* filename);
void writeVTKWithFixedBoundery(MyMesh &mesh, char* filename);

void readEleFile(MyTetMesh &mytet, std::string filename);
//����������vtk��ʽ
bool readTetVTK(MyTetMesh &mesh, std::string fname);

//�����ı�������
void readQuadVTK(PolyMesh &mesh, char* filename);

//����ı�������
void writeQuadVTK(PolyMesh &mesh, char* filename);


//��������������
void readTriVTK(MyMesh &mesh, char* filename);

//�������������
void writeTriVTK(MyMesh& mesh, char* filename);

//��������������
void readTritxt(MyMesh &mesh, char* filename1, char* filename2);

void readliu(MyMesh &mesh, char* filename);

#endif