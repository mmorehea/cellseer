function swc2obj(name)
    disp(name);
    f = importdata(name);
    fcont = f.data;
    disp('building obj cylinders')
    for ii = 1:size(fcont, 1)
        if fcont(ii, 7) == -1
            continue;
        %elseif fcont(ii, 2) == 1
            %continue;
        %elseif fcont(fcont(ii, 7), 2) == 1
            %continue;
        end
        
        ss = strcat('test/', num2str(ii), '.obj');
        [x,y,z] = points2cylinder([fcont(fcont(ii, 7), 6); fcont(ii, 6)], 10, fcont(fcont(ii, 7), 3:5), fcont(ii, 3:5));
        saveobjmesh(ss, x, y, z);
    end
    [status, ~] = system('chmod 777 ./tempobj/*');
    disp('status of chmod')
    disp(status)
    disp('flattening the objs')
    %[status, ~] = system('python flatten.py test');
    %disp(status);
    
    %movefile('test/out.obj', strcat(name, '.obj'));
    %delete('temp/out.obj');
end