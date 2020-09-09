%the function is to process all png img files in path
%./100_data/ : the source data
%./100_data/pre_250_250/ : after pre-process data,the result


path = './100_data/'; 
namelist = dir([path,'*.png']);
l = length(namelist);
for i = 1:l
    fprintf('%s',namelist(i).name);
    fprintf('\n');
    
    fprintf('start process----------');
    fprintf('\n');
    
    split_img(namelist(i).name) %only give img name to split_img(params)
    
    fprintf('end process----------');
    fprintf('\n');
end