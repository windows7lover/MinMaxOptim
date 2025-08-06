function [optimMeter, x, y] = alternatingDescentAscent(finfo, nIter, param)

[dim_x, dim_y, L_x, L_y, mu_x, mu_y, L_xy, f, gx, gy, x0, y0] = unpackFinfo(finfo);

optimMeter = OptimMeter(finfo, x0, y0, 'alt-GDA');

alpha = 0.5*min([1/L_x ; sqrt(mu_y)/(L_xy*sqrt(L_x))]);
beta = 0.5*min([1/L_y ; sqrt(mu_x)/(L_xy*sqrt(L_y))]);

x = x0;
y = y0;

numberGradientCall = 0;

while numberGradientCall<nIter
    x = x - alpha*gx(x,y); numberGradientCall = numberGradientCall+1;
    y = y + beta*gy(x,y); numberGradientCall = numberGradientCall+1;
    
    optimMeter = optimMeter.store(x, y, numberGradientCall);
    
end