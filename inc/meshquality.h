#ifndef _MESHQUALITYFILE_
#define _MESHQUALITYFILE_

#pragma once
#include <vector>
#include "dataType.h"

std::vector<double> computesizelenQuality(MyMesh& mesh);

std::vector<double> computeMeshAreaRatioQuality(MyMesh& mesh);
std::vector<double> computeMeshIdealElementQuality(MyMesh& mesh);
std::vector<double> computeMeshIdealElementQuality(MyTetMesh& mesh);

std::vector<double> computeMeshAngleQuality(MyMesh& mesh);
std::vector<double> computeMeshAngleQuality(MyTetMesh& mesh);
std::vector<double> computeMeshAngleQuality(PolyMesh& mesh);

double computeMeshAveang(MyMesh& mesh);

//输出网格质量
void outMeshIdealElementQuality(std::vector<double> &qualities);
void outMeshAngleQuality(std::vector<double> &qualities);
void outFileMeshAngleQuality(std::string fname, std::vector<double> &qualities2);
void outFileMeshIdealElementQuality(std::string fname, std::vector<double>& qualities2);


double computeLocalAngleQuality(MyMesh::Point p, std::vector<MyMesh::Point> ring);

bool isImprovedLocally(MyMesh &mesh, MyMesh::VertexHandle vh, MyMesh::Point &cog);
bool isImprovedLocally(MyTetMesh &mesh, OpenVolumeMesh::VertexHandle vh, Vec3d &cog);

#endif