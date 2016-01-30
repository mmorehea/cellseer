function bigMatrix = spinner(durr)


l = dir(durr);

%bigMatrix = zeros(length(l)*2000, 153);
bigMatrix = [];

for ii = 1:length(l)
    disp(l(ii).name)
    if length(l(ii).name) < 3
        continue
    end
    
    n = strcat(durr,'/',l(ii).name)
    if length(findstr(n, 'not')) == 0
        ff = importCSVfile(n);
        bigMatrix = vertcat(bigMatrix, ff);
        %bigMatrix(1+2002*(ii-1):2002*ii, :) = ff;
    end
            
end

    

