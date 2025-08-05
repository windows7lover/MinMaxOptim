% Obj function parameters

dim_x = 50;
dim_y = 75;

mu_x = 0.1;
mu_y = 0.01;

L_x = 10;
L_y = 100;

L_xy = 10;


% Generate Obj Function
finfo = randomQuadratic(dim_x, dim_y, L_x, L_y, mu_x, mu_y, L_xy);

% Algo Parameter
nIter = 10000;

param_GDA = [];


% Concatenate algo for experiment
algoCell = {...
    @(x,y,z) alternatingDescentAscent(x,y,z), param_GDA
    };
numExp = size(algoCell,1);

algoResult = cell(numExp,1);
finalIterate = cell(numExp,2);

% Run the experiment
for exp=1:size(numExp,1)
    [algoResult{exp}, finalIterate{exp,1}, finalIterate{exp,2}] = algoCell{exp,1}(finfo, nIter, algoCell{exp,2});
end

% Plot the experiment
fstar = finfo.fstar;

figure
for exp=1:numExp
    semilogy(algoResult{exp}.iterVec, algoResult{exp}.fval-fstar)
    hold on
end

figure
for exp=1:numExp
    semilogy(algoResult{exp}.iterVec, algoResult{exp}.gradval_x + algoResult{exp}.gradval_y)
    hold on
end