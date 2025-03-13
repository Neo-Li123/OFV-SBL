% 找到几何质心
function [center]=FindCentroid(Points)
if size(Points,1)>2
    Point_start=Points(1,:);
    Points=[Points;Point_start]; % 首尾闭合
    conv_index=convhull(Points);% 构建凸包
    Area=0;
    Center_x=0;
    Center_y=0;
    Point_x=Points(conv_index,1);
    Point_y=Points(conv_index,2);
    N=size(Point_x,1);
    for nn=1:N-1
        cross_term=Point_x(nn)*Point_y(nn+1)-Point_x(nn+1)*Point_y(nn);
        Area=Area+cross_term;
        Center_x=Center_x+(Point_x(nn)+Point_x(nn+1))*cross_term;
        Center_y=Center_y+(Point_y(nn)+Point_y(nn+1))*cross_term;
    end
    Area=0.5*Area;
    Center_x=Center_x/6/Area;
    Center_y=Center_y/6/Area;
    center=[Center_x,Center_y];
else
    center=[mean(Points(:,1)),mean(Points(:,2))];
end
