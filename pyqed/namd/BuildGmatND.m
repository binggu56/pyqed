function [gmat]=BuildGmatND(geom,mass,coord)
%[gmat]=BuildGmatND(geom,mass,coord)
%geom: input geometry (3,N)
%mass: masses according to geom (1,N) in [u]
%coord: Struct containing:
%       .ndims: number of dimensions
%       .dimn: (3,N) vector for n-th dimension
%
%gmat: G-matrix

%Convert mass tu a.u.
mass=mass.*1822.89;

%initialize inverse G-Matrix
gmat=zeros(coord.ndims,coord.ndims);

dq=0.001; %dq for derivative


for i=1:1:coord.ndims

    veci=eval(sprintf('coord.dim%d',i)); %get vector for dimension i

    %central difference derivative for dimension i
    geomi1=geom-dq.*veci;
    geomi2=geom+dq.*veci;
    dxdqi=(geomi1-geomi2)./(2*dq);
    for j=1:1:coord.ndims
        vecj=eval(sprintf('coord.dim%d',j)); %get vector for dimension j

        %central difference derivative for dimension j
        geomj1=geom-dq.*vecj;
        geomj2=geom+dq.*vecj;
        dxdqj=(geomj1-geomj2)./(2*dq);

        %Assign inverse  G-Matrix element
        gmat(i,j)=sum(mass*(dxdqi.*dxdqj)');
    end
end

gmat=inv(gmat); %invert to get G-matrix


end
