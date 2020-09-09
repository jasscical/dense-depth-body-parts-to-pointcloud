close all;
dataset = {'hard-pose'};
group = {'test'};
for datasetname = dataset
    datasetname1 = cell2mat(datasetname);
    fprintf(datasetname1);
    for group_name1 = group
        group_name = cell2mat(group_name1);
        fileFolder=fullfile(get_ref(datasetname1),group_name); 
        dirOutput=dir(fullfile(fileFolder)); 
        fileNames={dirOutput.name};
        for section_number=fileNames(3:end)
            frames_num = numel(dir(fullfile(fileFolder, cell2mat(section_number),'images','depthRender','Cam1','*.png'))); 
            for i = 1:frames_num
                savepath = fullfile('D:','hard_pose_processed_8197',group_name, cell2mat(section_number),num2str(i));
                if exist(savepath)==0   %该文件夹不存在，则直接创建
                    mkdir(savepath);
                else
                    if exist(fullfile(savepath,'skeleton.ply'))~=0
                        continue;
                    end;
                end;
                fprintf('%s\t%d\n',cell2mat(section_number),i);
                optional_frames = i:i;
                instances = load_multicam(datasetname1, group_name, cell2mat(section_number), optional_frames);

              %% The Raw Data

                % Each instance has the data from each camera and the grountruth posture.
                
                % Let's pick a random frame and show the raw data.
                for target_frame = optional_frames
                    instance = instances(target_frame);
                    
                    %% Processed data

                    % Now we need the depth map matrix from Kinect 2. This toolkit includes a
                    % copy in ./metadata/mapper.mat; The depth map is used to reconstruct 3d
                    % point cloud from the depth images.
       
                    map_file = load('mapper.mat');
                    mapper = map_file.mapper;
                    clear map_file;
                    
                    fprintf('start--------------');
                    [allp_sample, pose] = render2point_cloud(instance, mapper);
                    pcwrite(pointCloud(allp_sample),[savepath,'\','pointcloud.ply'],'PLYFormat','binary');
                    pcwrite(pointCloud(pose.joint_locations),[savepath,'\','skeleton.ply'],'PLYFormat','binary');
                    fprintf('end--------------');
                end;
            end;
        end;
    end;
end;
