clc;clear;
%%
addpath('D:\Codes\Mesh_Generation\ANN_AFT_tri\Optimize\')
TR = stlread("./6/sixth.stl");
triplot(TR);
axis equal;
axis off;
hold on;
%%
F = freeBoundary(TR);

% x = TR.Points(:,1);
% y = TR.Points(:,2);
% plot(x(F),y(F),'-r','LineWidth',2)
% axis equal;
% axis off;
%%
wallNodes = unique(F(:));
[xx, yy] = SpringOptimize(TR,[],wallNodes,3);

%%
% TR2 = delaunayTriangulation(x,y);
% figure
% triplot(TR2);
% axis equal;
% axis off;
% 
% [F2,P2] = freeBoundary(TR2); % 此时F中的编号已经与原先TR中的不一样了，而只与P对应
% x = P2(:,1);
% y = P2(:,2);
% plot(x(F2),y(F2),'-r','LineWidth',2)
% axis equal;
% axis off;