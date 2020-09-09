addpath(genpath('./models/'));
addpath(genpath('./ubc3vtoolkit'));

% Update this if you already have a matconvnet on your computer.
% It requires compilation.
run ./matconvnet/matlab/vl_setupnn.m

%% Load and prepare the image.
png_name = 'mayaProject.000983.png';
png_path = strcat('./5_data/', png_name);
save_path = strcat('./100_data/pre_250_250/' , png_name);

depth_image = importdata(png_path); %读取png_path为数组depth_image

% Find the boundaries of the person in the depth image.
margin_1 = sum(depth_image.alpha, 1); %depth_image.alpha的第 1 维度（列中像素值）相加，作为纵向（左右边界）margin_1
margin_2 = sum(depth_image.alpha, 2); %depth_image.alpha的第 2 维度（行中像素值） 相加，作为横向（上下边界）margin_2
%上下边界margin_2
top = find(margin_2, 1, 'first');%寻找margin_2前1个非0元素,上边界
bottom = find(margin_2, 1, 'last');%寻找margin_2倒数第1个非0元素，下边界
%左右边界margin_1
left = find(margin_1, 1, 'first');%寻找margin_1前1个非0元素，左边界
right = find(margin_1, 1, 'last');%右边界

assert( top<bottom && left<right, 'Top/Bottom & Left/Right are not consistent');
%人像的宽、高
width = right-left+1;
height = bottom-top+1;


% Crop the image
depth_image.cdata = depth_image.cdata(top:bottom, left:right, 1);
depth_image.alpha = depth_image.alpha(top:bottom, left:right);

target_window = 190;
margin = 30;
%缩放比例scale
if height > width,
    scaler = [target_window NaN]; %如果 高>宽，height放缩至target_window,width更小，放缩至target_window，
else                              %为跟height一样，超出width部分用 NaN 代替？？
    scaler = [NaN target_window];    
end

% Scale the image
depth_im = imresize(depth_image.cdata, scaler, 'lanczos2'); %放缩depth_image.cdata的scaler倍，改变图像尺寸时采用lanczos2算法；depth_im是类似待填充背景块？
depth_mask = imresize(depth_image.alpha, scaler, 'nearest') == 255; %即改变图像尺寸时采用最近邻插值算法；depth_mask是存储的alpha值，放缩后的alpha值等于255的才赋值给depth_mask

average_depth = mean(depth_im(depth_mask));%depth_im 中的 depth_mask平均深度
depth_im(depth_mask) = uint8(  int64(depth_im(depth_mask)) + int64(55-average_depth) ); %对depth_im 中的 depth_mask 进行操作
depth_im(~depth_mask) = 255; %对depth_im 中的 非 depth_mask 全部赋值为 255

depth_im = single(depth_im)/255;
[height, width] = size(depth_im);

final_depth_image = ones(target_window + 2*margin, target_window + 2*margin, 'single'); %是一个190+2*30 × 190+2*30的
final_depth_mask = zeros([target_window + 2*margin, target_window + 2*margin], 'uint8');
loc_x = floor((target_window + 2*margin - width)/2)+1;
loc_y = floor((target_window + 2*margin - height)/2)+1;

final_depth_image(loc_y:loc_y+height-1, loc_x:loc_x+width-1) = depth_im;
final_depth_mask(loc_y:loc_y+height-1, loc_x:loc_x+width-1) = depth_mask*255;

%% Load and prepare the network.
net = load('hardpose_69k_dag');
net = dagnn.DagNN.loadobj(net);

net.eval({'data', final_depth_image});

prediction = net.getVar('prob').value;
%% Visualize the predcitions

[~, classes] = max(prediction, [], 3);%返回Prediction中第3维度中最大维度给classes
classes(~final_depth_mask) = 0;
imagesc(classes);

MappedData = mapminmax(classes, 0, 1);%数据规一化

% filename = 'G:/dense-depth-body-parts-master/classes.txt';
% dlmwrite(filename,classes);


%% Save the classes image
imwrite(MappedData,save_path,'png');
fprintf('end----------------');



% set(gca,'XTick',[]) % Remove the ticks in the x axis!
% set(gca,'YTick',[]) % Remove the ticks in the y axis
% set(gca,'Position',[0 0 1 1]) % Make the axes occupy the hole figure
% saveas(gcf,png_path,'png')



