function swc2obj(name)
    fcont = loadSWC(name);
    disp('building obj cylinders')
    for ii = 1:size(fcont, 1)
        if fcont(ii, 7) == -1
            continue;
        %elseif fcont(ii, 2) == 1
            %continue;
        %elseif fcont(fcont(ii, 7), 2) == 1
            %continue;
        end
        
        ss = strcat('tempobj/', num2str(ii), '.obj');
        [x,y,z] = points2cylinder([fcont(fcont(ii, 7), 6); fcont(ii, 6)], 10, fcont(fcont(ii, 7), 3:5), fcont(ii, 3:5));
        saveobjmesh(ss, x, y, z);
    end
%     [status, ~] = system('chmod 777 ./tempobj/*');
%     disp('status of chmod')
%     disp(status)
%     disp('flattening the objs')
    %[status, ~] = system('python flatten.py test');
    %disp(status);
    
    %movefile('test/out.obj', strcat(name, '.obj'));
    %delete('temp/out.obj');
end

function saveobjmesh(name,x,y,z,nx,ny,nz)
% SAVEOBJMESH Save a x,y,z mesh as a Wavefront/Alias Obj file
% SAVEOBJMESH(fname,x,y,z,nx,ny,nz)
%     Saves the mesh to the file named in the string fname
%     x,y,z are equally sized matrices with coordinates.
%     nx,ny,nz are normal directions (optional)

  
  normals=1;
  if (nargin<5) normals=0; end
  l=size(x,1); h=size(x,2);  

  n=zeros(l,h);
  fid=fopen(name,'w');
  nn=1;
  for i=1:l
    for j=1:h
      n(i,j)=nn; 
      fprintf(fid, 'v %f %f %f\n',x(i,j),y(i,j),z(i,j)); 
      fprintf(fid, 'vt %f %f\n',(i-1)/(l-1),(j-1)/(h-1)); 
      if (normals) fprintf(fid, 'vn %f %f %f\n', nx(i,j),ny(i,j),nz(i,j)); end
      nn=nn+1;
    end
  end
  fprintf(fid,'g mesh\n');
  
  for i=1:(l-1)
    for j=1:(h-1)
      if (normals) 
	fprintf(fid,'f %d/%d/%d %d/%d/%d %d/%d/%d %d/%d/%d\n',n(i,j),n(i,j),n(i,j),n(i+1,j),n(i+1,j),n(i+1,j),n(i+1,j+1),n(i+1,j+1),n(i+1,j+1),n(i,j+1),n(i,j+1),n(i,j+1));
      else
  	fprintf(fid,'f %d/%d %d/%d %d/%d %d/%d\n',n(i,j),n(i,j),n(i+1,j),n(i+1,j),n(i+1,j+1),n(i+1,j+1),n(i,j+1),n(i,j+1));
      end
    end
  end
  fprintf(fid,'g\n\n');
  fclose(fid);
end

%  CYLINDER:  A function to draw a N-sided cylinder based on the
%             generator curve in the vector R.
%
%  Usage:      [X, Y, Z] = cylinder(R, N)
%
%  Arguments:  R - The vector of radii used to define the radius of
%                  the different segments of the cylinder.
%              N - The number of points around the circumference.
%
%  Returns:    X - The x-coordinates of each facet in the cylinder.
%              Y - The y-coordinates of each facet in the cylinder.
%              Z - The z-coordinates of each facet in the cylinder.
%
%  Author:     Luigi Barone
%  Date:       9 September 2001
%  Modified:   Per Sundqvist July 2004
function [X, Y, Z] = points2cylinder(R, N,r1,r2)

    theta = linspace(0,2*pi,N);

    m = length(R);                 % Number of radius values
                                   % supplied.

    if m == 1                      % Only one radius value supplied.
        R = [R; R];                % Add a duplicate radius to make
        m = 2;                     % a cylinder.
    end


    X = zeros(m, N);             % Preallocate memory.
    Y = zeros(m, N);
    Z = zeros(m, N);
    
    v=(r2-r1)/sqrt((r2-r1)*(r2-r1)');    %Normalized vector;
    %cylinder axis described by: r(t)=r1+v*t for 0<t<1
    R2=rand(1,3);              %linear independent vector (of v)
    x2=v-R2/(R2*v');    %orthogonal vector to v
    x2=x2/sqrt(x2*x2');     %orthonormal vector to v
    x3=cross(v,x2);     %vector orthonormal to v and x2
    x3=x3/sqrt(x3*x3');
    
    r1x=r1(1);r1y=r1(2);r1z=r1(3);
    r2x=r2(1);r2y=r2(2);r2z=r2(3);
    vx=v(1);vy=v(2);vz=v(3);
    x2x=x2(1);x2y=x2(2);x2z=x2(3);
    x3x=x3(1);x3y=x3(2);x3z=x3(3);
    
    time=linspace(0,1,m);
    for j = 1 : m
      t=time(j);
      X(j, :) = r1x+(r2x-r1x)*t+R(j)*cos(theta)*x2x+R(j)*sin(theta)*x3x; 
      Y(j, :) = r1y+(r2y-r1y)*t+R(j)*cos(theta)*x2y+R(j)*sin(theta)*x3y; 
      Z(j, :) = r1z+(r2z-r1z)*t+R(j)*cos(theta)*x2z+R(j)*sin(theta)*x3z;
    end

    %surf(X, Y, Z);
end

function a = loadSVC(filename, b_minusFirst)

if nargin<2,
    b_minusFirst=0;
end;

L = loadfilelist(filename);
a = zeros(length(L), 7);

k=0;
for i=1:length(L),
    if isempty(deblank(L{i})),
        continue;
    end;
    if (L{i}(1)=='#'),
        continue;
    end;
    
    k=k+1;
    tmp = str2num(L{i});
    a(k,:) = tmp(1:7);
end;

a = a(1:k,:); %%remove the non-used lines

%make sure all the origin (neuron soma) will be 0
if b_minusFirst,
    a(:,3:5) = a(:,3:5) - repmat(a(1,3:5), size(a,1), 1);
end;

return;
end

function filelist = loadfilelist(filename)

filelist = [];
fid = fopen(filename);
if fid==-1,
    disp(['Error to open the file : ' filename]);
    return;
else
    i=1;
    while 1
        tline = fgetl(fid);
        if ~ischar(tline), break; end;
        filelist{i} = deblank(tline);
        i = i+1;
    end;
end;
fclose(fid);
end