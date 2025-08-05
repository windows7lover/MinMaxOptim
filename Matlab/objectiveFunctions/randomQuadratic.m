% Random Quadratic

function finfo = randomQuadratic(dim_x, dim_y, L_x, L_y, mu_x, mu_y, L_xy)

finfo = struct( ...
    'dim_x', dim_x, ...
    'dim_y', dim_y, ...
    'L_x', L_x, ...
    'L_y', L_y, ...
    'mu_x', mu_x, ...
    'mu_y', mu_y, ...
    'L_xy', L_xy ...
);

matrix_x = random_psd_matrix(dim_x, mu_x, L_x);
matrix_y = random_psd_matrix(dim_y, mu_y, L_y);

linear_x = randn(dim_x,1);
linear_y = randn(dim_y,1);

matrix_xy = randn(dim_x,dim_y);
matrix_xy  = L_xy*matrix_xy/norm(matrix_xy);

finfo.f = @(x, y) x'*matrix_x*x/2 + x'*matrix_xy*y - y'*matrix_y*y/2 + linear_x'*x + linear_y'*y;
finfo.gx = @(x, y) matrix_x*x + matrix_xy*y + linear_x;
finfo.gy = @(x, y) matrix_xy'*x - matrix_y*y + linear_y;

finfo.x0 = randn(dim_x,1);
finfo.y0 = randn(dim_y,1);

solution = -[matrix_x matrix_xy ; matrix_xy' -matrix_y]\[linear_x;linear_y];

finfo.xstar = solution(1:dim_x);
finfo.ystar = solution(dim_x+1:end);
finfo.fstar = finfo.f(finfo.xstar, finfo.ystar);