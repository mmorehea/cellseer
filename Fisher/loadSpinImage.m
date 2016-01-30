function [picked ,nonpicked] = loadSpinImage(fName, mode)
if mode
   picked = csvread(fName+'picked.csv');
   nonpicked  = [];
else 
   picked = csvread(fName+'picked.csv');
   nonpicked = csvread(fName+'notpicked.csv');
end