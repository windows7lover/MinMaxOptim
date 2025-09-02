function [optimMeter, x, y] = approxMinMax(finfo, nIter, param)

[dim_x, dim_y, L_x, L_y, mu_x, mu_y, L_xy, f, gx, gy, x0, y0] = unpackFinfo(finfo);

optimMeter = OptimMeter(finfo, x0, y0, 'approxMinMax');

x = x0;
y = y0;

xstar = x0;
ystar = y0;

numberGradientCall = 0;

factor_w = 0.25;

hx = 1/(L_x + L_xy^2/mu_y);
hy = 1/(L_y + L_xy^2/mu_x);

Lyapunov = @(x,y,xstar, ystar) f(x,ystar)-f(xstar,y)+norm(gx(x,ystar))^2/mu_x+norm(gx(xstar,y))^2/mu_y;


while numberGradientCall<nIter
    
    
    delta_w = @(xplus, ystar) f(xplus,ystar)-f(x,ystar) + (norm(gy(xplus,ystar))^2-norm(gy(x,ystar))^2)/(2*mu_y);
    
    loop1 = numberGradientCall;
    
    Lyapunov(x,y,xstar, ystar)
    condition_x_satisfied = false;
    while ~condition_x_satisfied
        
        ystar = ystar + gy(x,ystar)/L_y; numberGradientCall = numberGradientCall+1;
        xplus = x - hx*gx(x,ystar); numberGradientCall = numberGradientCall+1;
        condition_x_satisfied = (delta_w(xplus, ystar) <= factor_w * (-0.5 * norm(gx(x,ystar))^2)*hx);
    end
    x = xplus;
    
    
    % We are ready to minimize w(x)
    anchor_x = x;
    for i=1:1
        grad_w = gx(x,ystar)+L_xy^2*(x-anchor_x)/mu_y;
        x = x - hx*(grad_w);  numberGradientCall = numberGradientCall+1;
       norm(grad_w)
    end
    
%     loop1 = numberGradientCall-loop1;
    Lyapunov(x,y,xstar, ystar)
    
%     [loop1, norm(gx(x,ystar)), norm(gy(x,ystar)), Lyapunov(x,y,xstar, ystar)]
    
    
    delta_v = @(yplus, xstar) -f(xstar,yplus)+f(xstar,y) + (norm(gx(xstar,yplus))^2-norm(gx(xstar,y))^2)/(2*mu_x);
    
    loop2 = numberGradientCall;
    condition_y_satisfied = false;
    while ~condition_y_satisfied
        xstar = xstar - gx(xstar,y)/L_x; numberGradientCall = numberGradientCall+1;
        yplus = y + hy*gy(xstar,y); numberGradientCall = numberGradientCall+1;
        condition_y_satisfied = (delta_v(yplus, xstar) <= factor_w * (-0.5 * norm(gy(xstar,y))^2)*hy);
    end
    y = yplus;
    loop2 = numberGradientCall-loop2;
    
    
    % We are ready to minimize v(x)
    Lyapunov(x,y,xstar, ystar)
    anchor_y = y;
    for i=1:1
        grad_v = -gy(xstar,y)+L_xy^2*(y-anchor_y)/mu_x;
        y = y - hy*(grad_v);  numberGradientCall = numberGradientCall+1;
       norm(grad_v)
    end
    
    Lyapunov(x,y,xstar, ystar)
%     [loop1, loop2]
    % pause
    
    optimMeter = optimMeter.store(x, y, numberGradientCall);
    
end