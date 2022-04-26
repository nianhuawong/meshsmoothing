#ifndef _WRITEFILE_
#define _WRITEFILE_
#pragma once
#include "dataType.h"

// ----------------------------------------------------------------------------

void writeVTK(MyMesh& mesh, char* filename);
void writeVTKWithFixedBoundery(MyMesh &mesh, char* filename);

void readEleFile(MyTetMesh &mytet, std::string filename);
//输入四面体vtk格式
bool readTetVTK(MyTetMesh &mesh, std::string fname);

//输入四边形网格
void readQuadVTK(PolyMesh &mesh, char* filename);

//输出四边形网格
void writeQuadVTK(PolyMesh &mesh, char* filename);


//输入三角形网格
void readTriVTK(MyMesh &mesh, char* filename);

//输出三角形网格
void writeTriVTK(MyMesh& mesh, char* filename);

//输入三角形网格
void readTritxt(MyMesh &mesh, char* filename1, char* filename2);

void readliu(MyMesh &mesh, char* filename);

#endif