function buildOBJ(path)

F = loadSWC(path);


for ii = 1:size(F, 1)
    if F(ii, 7) == -1
        continue
    end
    
    [x,y,z] = points2cylinder(.001, 10, F(ii, 3:5), F(F(ii, 7), 3:5));
    ss = strcat('./test/',num2str(ii), '.obj');
    saveobjmesh(ss, x, y, z);
            
        end
        

end
    





    