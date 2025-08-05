function [optimMeter, x, y] = alternatingDescentAscent(finfo, nIter, param)

[dim_x, dim_y, L_x, L_y, mu_x, mu_y, L_xy, f, gx, gy, x0, y0] = unpackFinfo(finfo);

optimMeter = OptimMeter(finfo, x0, y0, 'alt-GDA');

alpha = 0.5*min([1/L_x ; sqrt(mu_y)/(L_xy*sqrt(L_x))]);
beta = 0.5*min([1/L_y ; sqrt(mu_x)/(L_xy*sqrt(L_y))]);

x = x0;
y = y0;

for i=1:nIter
    x = x - alpha*gx(x,y);
    y = y + beta*gy(x,y);
    
    optimMeter = optimMeter.store(x, y, i);
    
end