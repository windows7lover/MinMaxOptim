function [optimMeter, x, y] = approxMinMax(finfo, nIter, param)

[dim_x, dim_y, L_x, L_y, mu_x, mu_y, L_xy, f, gx, gy, x0, y0] = unpackFinfo(finfo);

optimMeter = OptimMeter(finfo, x0, y0, 'approxMinMax');

alpha = 2/(mu_x+L_x);
beta = 0.5*min([1/L_y ; sqrt(mu_x)/(L_xy*sqrt(L_y))]);

x = x0;
y = y0;

numberGradientCall = 0;

while numberGradientCall<nIter
    
    for j=1:10
        x = x - alpha*gx(x,y); numberGradientCall = numberGradientCall+1;
    end
    
    for j=1:10
        y = y + beta*gy(x,y); numberGradientCall = numberGradientCall+1;
    end
    
    optimMeter = optimMeter.store(x, y, numberGradientCall);
    
end