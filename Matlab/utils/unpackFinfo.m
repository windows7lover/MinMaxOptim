function [dim_x, dim_y, L_x, L_y, mu_x, mu_y, L_xy, f, gx, gy, x0, y0] = unpackFinfo(finfo)
%UNPACKFINFO Extract specific fields from a finfo struct.
%
%   [dim_x, dim_y, L_x, L_y, mu_x, mu_y, L_xy, f, gx, gy, x0, y0] = unpackFinfo(finfo)
%
%   This function returns the main problem parameters from a structured input.

    dim_x = finfo.dim_x;
    dim_y = finfo.dim_y;
    L_x   = finfo.L_x;
    L_y   = finfo.L_y;
    mu_x  = finfo.mu_x;
    mu_y  = finfo.mu_y;
    L_xy  = finfo.L_xy;
    f     = finfo.f;
    gx    = finfo.gx;
    gy    = finfo.gy;
    x0    = finfo.x0;
    y0    = finfo.y0;
end
