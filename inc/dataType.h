#pragma once
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <OpenVolumeMesh/Mesh/TetrahedralMesh.hh>
#include <OpenVolumeMesh/Mesh/PolyhedralMesh.hh>
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>
#include <OpenVolumeMesh/Geometry/VectorT.hh>
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/Handles.hh>
#include <OpenMesh/Tools/Smoother/JacobiLaplaceSmootherT.hh>
#include <OpenMesh/Tools/Smoother/LaplaceSmootherT.hh>

#define PI 3.14159265

typedef OpenMesh::TriMesh_ArrayKernelT<>  MyMesh;

typedef OpenMesh::PolyMesh_ArrayKernelT<>  PolyMesh;

typedef OpenVolumeMesh::GeometricTetrahedralMeshV3d MyTetMesh;

typedef OpenVolumeMesh::Geometry::Vec3d         Vec3d;
typedef OpenVolumeMesh::Geometry::Vec3f         Vec3f;
typedef OpenVolumeMesh::GeometryKernel<Vec3f>   PolyhedralMeshV3f;





