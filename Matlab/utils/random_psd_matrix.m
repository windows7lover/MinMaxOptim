function matrix = random_psd_matrix(dim, mu, L)

[randQ,~] = qr(randn(dim));
matrix = randQ * diag([ mu ; mu + rand(dim-2,1)*(L-mu) ; L]) * randQ';
