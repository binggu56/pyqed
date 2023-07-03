rdir = 'D:\uci_files\thio-uracil\';

[s0min,mass] = ReadXyz([rdir 's0min.xyz'],2);
s0min = s0min./0.5291772083;
s2min = ReadXyz([rdir 's2min.xyz'],2);
s2min = SatisfyEckart(s0min,s2min,mass(1,:))./0.5291772083;
% s2min = SatisfyEckart(s0min,s2min,mass(1,:));
coin = ReadXyz([rdir 's2s1coin.xyz'],2);
coin = SatisfyEckart(s0min,coin,mass(1,:))./0.5291772083;
% coin = SatisfyEckart(s0min,coin,mass(1,:));

v1 = coin - s0min;
v1 = v1 / sqrt( sum(sum(v1.*v1)) );

v2 = s2min - s0min;
v2 = v2 - sum(sum(v2.*v1))*v1;
v2 = v2 / sqrt( sum(sum(v2.*v2)) );

coord=struct;
coord.ndims = 2;
coord.dim1 = v1;
coord.dim2 = v2;

gmatu = BuildGmatND(s0min,mass(1,:),coord);


%% Points necessary in each dimension
q1 = (-0.5:0.1:1)./0.5291772083;
q2 = (-0.5:0.1:0.5)./0.5291772083;

Tmax = 3.5 ;                % potential energy gained by the wp [eV] x 1.5
Tmax = Tmax / 27.21;        % [eV] to [au]
k1m = sqrt((2*Tmax) ./ ( gmatu(1,1) - gmatu(1,2).^2./gmatu(2,2) ));
k2m = sqrt((2*Tmax) ./ ( gmatu(2,2) - gmatu(1,2).^2./gmatu(1,1) ));
dx1 = pi / k1m;
dx2 = pi / k2m;
Nx1 = (((q1(end)-q1(1))) / dx1) + 1
Nx2 = (((q2(end)-q2(1))) / dx2) + 1
