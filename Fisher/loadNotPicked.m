function [encodings, names] = loadNotPicked(durr, means, priors, covariances)


l = dir(durr);

%bigMatrix = zeros(length(l)*2000, 153);
encodings = [];
names = {};
for ii = 1:length(l)
    disp(l(ii).name)
    if length(l(ii).name) < 3
        continue
    end
    
    n = strcat(durr,'/',l(ii).name);
    if length(findstr(n, 'not')) > 0
        disp(n)
        ff = importCSVfile(n);
        enc = vl_fisher(ff', means, covariances, priors);
        encodings = vertcat(encodings, enc');
        names = vertcat(names, l(ii).name);
        %bigMatrix(1+2002*(ii-1):2002*ii, :) = ff;
    end
            
end

save('fisher_vector', 'encodings', 'names')

