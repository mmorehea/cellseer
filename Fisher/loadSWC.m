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